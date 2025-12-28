import numpy as np
import polars as pl
import streamlit as st
from mplsoccer import Pitch
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -------------------------
# Small shared utils
# -------------------------
def require_cols(df: pl.DataFrame, cols: list[str], label: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        st.error(f"Missing required columns in {label}: {missing}")
        st.stop()

def first_existing(df: pl.DataFrame, cols: list[str]):
    for c in cols:
        if c in df.columns:
            return c
    return None

def safe_bool_expr(expr: pl.Expr) -> pl.Expr:
    return (
        expr.cast(pl.Utf8)
        .str.to_lowercase()
        .is_in(["true", "1", "yes", "y"])
    )

def fig_height(n_rows: int, base: float = 4.0, per_row: float = 0.35) -> float:
    return max(base, per_row * max(1, n_rows))

def stack_barh(ax, y, series_list, labels):
    left = np.zeros(len(y))
    for s, lab in zip(series_list, labels):
        ax.barh(y, s, left=left, label=lab)
        left += np.asarray(s)

def frame_end_int_pl() -> pl.Expr:
    return pl.coalesce([pl.col("frame_end"), pl.col("frame_start")]).cast(pl.Int64)

# =========================================================
# DATA BUILDERS (POLARS)
# =========================================================
def build_pressure_profile_pl(obe: pl.DataFrame) -> pl.DataFrame:
    require_cols(
        obe,
        [
            "event_id", "player_name", "team_shortname",
            "pressing_chain", "index_in_pressing_chain", "pressing_chain_index",
            "force_backward", "stop_possession_danger", "reduce_possession_danger"
        ],
        "OBE (Pressure Profile)"
    )

    df = obe.with_columns(
        pl.when(pl.col("pressing_chain") == True)
        .then(pl.lit("pressing_chain"))
        .otherwise(pl.lit("individual_pressure"))
        .alias("pressure_category")
    ).with_columns(
        safe_bool_expr(pl.col("force_backward")).cast(pl.Int8).alias("force_backward"),
        safe_bool_expr(pl.col("stop_possession_danger")).cast(pl.Int8).alias("stop_possession_danger"),
        safe_bool_expr(pl.col("reduce_possession_danger")).cast(pl.Int8).alias("reduce_possession_danger"),
    )

    chain_obe = df.filter((pl.col("pressing_chain") == True) & pl.col("pressing_chain_index").is_not_null())

    team_chain_counts = (
        chain_obe
        .select(["team_shortname", "pressing_chain_index"])
        .unique()
        .group_by("team_shortname")
        .agg(pl.count().alias("team_total_chains"))
        .rename({"team_shortname": "team"})
    )

    player_chain_inits = (
        chain_obe
        .filter(pl.col("index_in_pressing_chain") == 1)
        .select(["player_name", "team_shortname", "pressing_chain_index"])
        .unique()
        .group_by(["player_name", "team_shortname"])
        .agg(pl.count().alias("press_initiations"))
        .rename({"team_shortname": "team"})
    )

    player_volume = (
        df.group_by("player_name")
        .agg(
            pl.count().alias("total_engagements"),
            pl.col("pressure_category").eq("individual_pressure").sum().alias("individual_pressure"),
            pl.col("pressure_category").eq("pressing_chain").sum().alias("pressing_chain"),
        )
    )

    effectiveness = (
        df.group_by("player_name")
        .agg(
            pl.sum("force_backward").alias("force_backward"),
            pl.sum("stop_possession_danger").alias("stop_danger"),
            pl.sum("reduce_possession_danger").alias("reduce_danger"),
        )
    )

    player_teams = (
        df.group_by("player_name")
        .agg(pl.first("team_shortname").alias("team"))
    )

    profile = (
        player_volume
        .join(effectiveness, on="player_name", how="left")
        .join(player_teams, on="player_name", how="left")
        # preserve original behaviour: merge on player_name only
        .join(player_chain_inits.select(["player_name", "press_initiations"]), on="player_name", how="left")
        .join(team_chain_counts, on="team", how="left")
        .with_columns(
            pl.col("press_initiations").fill_null(0).cast(pl.Int64),
            pl.col("team_total_chains").fill_null(0).cast(pl.Int64),
        )
        .with_columns(
            pl.when(pl.col("team_total_chains") > 0)
            .then(pl.col("press_initiations") / pl.col("team_total_chains"))
            .otherwise(pl.lit(0.0))
            .alias("press_setter_share")
        )
    )

    final_columns = [
        "team",
        "total_engagements",
        "individual_pressure",
        "pressing_chain",
        "press_initiations",
        "team_total_chains",
        "press_setter_share",
        "force_backward",
        "stop_danger",
        "reduce_danger",
    ]
    return profile.select(["player_name"] + final_columns)


def build_press_breaks_pl(
    df: pl.DataFrame,
    fps: int,
    chain_gap_s: float,
    eligible_radius_m: float,
    end_window_s: float,
) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    obe = df.filter(pl.col("event_type") == "on_ball_engagement")
    pp = df.filter(pl.col("event_type") == "player_possession")

    require_cols(
        obe,
        ["pressing_chain", "pressing_chain_index", "frame_start", "frame_end",
         "team_id", "team_shortname", "player_id", "player_name"],
        "OBE (Press Breaks)"
    )
    require_cols(pp, ["frame_start", "team_id", "team_in_possession_phase_type"], "PP (Press Breaks)")

    dist_col = first_existing(
        obe,
        ["interplayer_distance_min", "interplayer_distance_start", "interplayer_distance_start_physical"]
    )
    if dist_col is None:
        return pl.DataFrame(), pl.DataFrame(), {"error": "No usable OBE distance column (expected interplayer_distance_*)."}

    goalside_col = first_existing(obe, ["goal_side_end", "goal_side_start"])
    traj_col = first_existing(obe, ["trajectory_direction"])
    overload_col = first_existing(obe, ["simultaneous_defensive_engagement_same_target"])

    chain_obe = (
        obe.filter((pl.col("pressing_chain") == True) & pl.col("pressing_chain_index").is_not_null())
        .with_columns(pl.col("pressing_chain_index").cast(pl.Int64))
    )
    if chain_obe.height == 0:
        return pl.DataFrame(), pl.DataFrame(), {"error": "No pressing chains found."}

    last_obe = (
        chain_obe.sort(["pressing_chain_index", "frame_start"])
        .group_by("pressing_chain_index")
        .agg(pl.all().last())
        .select(["pressing_chain_index", "team_id", "team_shortname", "frame_start", "frame_end"])
    )

    pp_sorted = pp.sort("frame_start")
    chain_gap_frames = int(chain_gap_s * fps)
    end_window_frames = int(end_window_s * fps)

    chain_obe2 = chain_obe.with_columns(frame_end_int_pl().alias("_frame_end_i"))

    rows = []
    for r in last_obe.to_dicts():
        chain_idx = int(r["pressing_chain_index"])
        defending_team_id = r["team_id"]

        chain_end = int(r["frame_end"]) if r["frame_end"] is not None else int(r["frame_start"])
        win_start, win_end = chain_end, chain_end + chain_gap_frames

        cand_pp = pp_sorted.filter(
            (pl.col("team_id") != defending_team_id) &
            (pl.col("frame_start") >= win_start) &
            (pl.col("frame_start") <= win_end) &
            (pl.col("team_in_possession_phase_type").is_in(["build_up", "create", "direct"]))
        )
        if cand_pp.height == 0:
            continue
        pp_phase = cand_pp.select(pl.col("team_in_possession_phase_type").first()).item()

        in_chain = chain_obe2.filter(pl.col("pressing_chain_index") == chain_idx)
        cand = in_chain.filter(
            (pl.col("_frame_end_i") >= (chain_end - end_window_frames)) &
            (pl.col("_frame_end_i") <= chain_end)
        )
        if cand.height == 0:
            cand = in_chain

        cand = cand.filter(pl.col(dist_col).is_not_null()).with_columns(pl.col(dist_col).cast(pl.Float64))
        cand = cand.filter(pl.col(dist_col) <= float(eligible_radius_m))
        if cand.height == 0:
            continue

        if goalside_col is not None:
            cand = cand.filter(safe_bool_expr(pl.col(goalside_col)))
            if cand.height == 0:
                continue

        if traj_col is not None:
            cand = cand.with_columns(pl.col(traj_col).cast(pl.Utf8).str.to_lowercase().alias("_traj_l"))
            cand = cand.filter(pl.col("_traj_l") != "backward")
            if cand.height == 0:
                continue

        if overload_col is not None:
            cand = cand.filter(~safe_bool_expr(pl.col(overload_col)))
            if cand.height == 0:
                continue

        cand = cand.sort([dist_col, "frame_start"], descending=[False, True])
        best = cand.head(1).to_dicts()[0]

        rows.append({
            "pressing_chain_index": chain_idx,
            "defending_team": best.get("team_shortname"),
            "defending_team_id": best.get("team_id"),
            "player_id": int(best.get("player_id")) if best.get("player_id") is not None else None,
            "player_name": best.get("player_name"),
            "phase": pp_phase,
            "distance_used": float(best.get(dist_col)),
            "distance_col": dist_col,
        })

    press_breaks = pl.DataFrame(rows)
    meta = {"dist_col": dist_col, "goalside_col": goalside_col, "traj_col": traj_col, "overload_col": overload_col}

    if press_breaks.height == 0:
        return press_breaks, pl.DataFrame(), meta

    rank_df = (
        press_breaks
        .group_by(["defending_team", "player_id", "player_name"])
        .agg(
            pl.count().alias("press_break_failures"),
            pl.col("phase").value_counts().alias("_phase_counts")
        )
        .with_columns(
            pl.col("_phase_counts")
            .map_elements(
                lambda lst: ", ".join([d["phase"] for d in (lst[:3] if isinstance(lst, list) else [])]),
                return_dtype=pl.Utf8
            )
            .alias("phases")
        )
        .drop("_phase_counts")
        .sort("press_break_failures", descending=True)
    )

    return press_breaks, rank_df, meta


def build_pressure_clusters_2d_pl(
    profile_main: pl.DataFrame,
    k: int = 3,
    min_engagements: int = 5,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    req = [
        "player_name",
        "total_engagements",
        "pressing_chain",
        "force_backward",
        "stop_danger",
        "reduce_danger",
    ]
    require_cols(profile_main, req, "Pressure Profile (Clustering 2D)")

    df = profile_main.filter(pl.col("total_engagements").cast(pl.Float64) >= float(min_engagements))
    if df.height == 0 or df.height < k:
        return pl.DataFrame(), pl.DataFrame()

    df = df.with_columns(
        pl.col("total_engagements").cast(pl.Float64).alias("_te"),
        pl.col("pressing_chain").cast(pl.Float64).alias("_pc"),
        pl.col("force_backward").cast(pl.Float64).alias("_fb"),
        pl.col("stop_danger").cast(pl.Float64).alias("_sd"),
        pl.col("reduce_danger").cast(pl.Float64).alias("_rd"),
    ).with_columns(
        pl.when(pl.col("_te") > 0).then((pl.col("_pc") / pl.col("_te")).clip(0, 1)).otherwise(0.0).alias("pressure_style"),
        pl.when(pl.col("_te") > 0).then((pl.col("_fb") + pl.col("_sd") + pl.col("_rd")) / pl.col("_te")).otherwise(0.0).alias("defensive_score"),
    )

    X = df.select(["pressure_style", "defensive_score"]).to_numpy()
    Xs = StandardScaler().fit_transform(X)

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_id = km.fit_predict(Xs).astype(int)

    df2 = df.with_columns(pl.Series("cluster_id", cluster_id))

    cluster_summary = (
        df2.group_by("cluster_id")
        .agg(
            pl.mean("pressure_style").alias("pressure_style"),
            pl.mean("defensive_score").alias("defensive_score"),
            pl.mean("total_engagements").alias("avg_total_engagements"),
        )
        .with_columns(
            pl.col("pressure_style").round(3),
            pl.col("defensive_score").round(3),
            pl.col("avg_total_engagements").round(3),
        )
        .sort("cluster_id")
    )

    rank = (
        cluster_summary.select(["cluster_id", "defensive_score"])
        .sort("defensive_score", descending=True)
        .to_dicts()
    )
    if k == 3 and len(rank) == 3:
        name_map = {
            int(rank[0]["cluster_id"]): "Effective Pressers",
            int(rank[1]["cluster_id"]): "System Contributors",
            int(rank[2]["cluster_id"]): "Low-Impact Pressers",
        }
    else:
        name_map = {int(r["cluster_id"]): f"Cluster {i+1}" for i, r in enumerate(rank)}

    df2 = df2.with_columns(
        pl.col("cluster_id").map_elements(lambda x: name_map.get(int(x), f"Cluster {int(x)}"), return_dtype=pl.Utf8).alias("cluster_name")
    )
    cluster_summary = cluster_summary.with_columns(
        pl.col("cluster_id").map_elements(lambda x: name_map.get(int(x), f"Cluster {int(x)}"), return_dtype=pl.Utf8).alias("cluster_name")
    )

    out_cols = [
        "player_name",
        "pressure_style",
        "defensive_score",
        "total_engagements",
        "cluster_id",
        "cluster_name",
    ]
    player_df = df2.select(out_cols).sort(["cluster_name", "player_name"])
    return player_df, cluster_summary


# =========================================================
# PLOTS (keep matplotlib/mplsoccer; input = pandas or polars where noted)
# =========================================================
def plot_engagement_split(profile_pd, title_team: str):
    plot_df = profile_pd[
        ["player_name", "pressing_chain", "individual_pressure", "press_initiations", "total_engagements"]
    ].sort_values("total_engagements", ascending=False)

    fig, ax = plt.subplots(figsize=(10, fig_height(len(plot_df))))
    y = plot_df["player_name"]

    stack_barh(
        ax,
        y=y,
        series_list=[plot_df["pressing_chain"], plot_df["individual_pressure"]],
        labels=["pressing_chain", "individual_pressure"],
    )

    ax.scatter(
        plot_df["press_initiations"],
        y,
        marker="o",
        s=60,
        label="press initiations (unique chains)"
    )

    ax.invert_yaxis()
    ax.set_xlabel("engagements (bars) / initiated chains (dots)")
    ax.set_ylabel("player_name")
    ax.set_title(f"Engagement split & press initiations — {title_team}")
    ax.legend()
    st.pyplot(fig, clear_figure=True)


def plot_press_outcomes(profile_pd, title_team: str):
    plot_df = profile_pd[["player_name", "force_backward", "stop_danger", "reduce_danger"]].copy()
    plot_df["total_outcomes"] = plot_df["force_backward"] + plot_df["stop_danger"] + plot_df["reduce_danger"]
    plot_df = plot_df.sort_values("total_outcomes", ascending=False)

    fig, ax = plt.subplots(figsize=(10, fig_height(len(plot_df))))
    y = plot_df["player_name"]

    stack_barh(
        ax,
        y=y,
        series_list=[plot_df["force_backward"], plot_df["stop_danger"], plot_df["reduce_danger"]],
        labels=["force_backward", "stop_danger", "reduce_danger"],
    )

    ax.invert_yaxis()
    ax.set_xlabel("outcome counts (OBE flags summed)")
    ax.set_ylabel("player_name")
    ax.set_title(f"Press outcomes — {title_team}")
    ax.legend()
    st.pyplot(fig, clear_figure=True)


def plot_press_setter_share(pss_pd, team: str):
    fig, ax = plt.subplots(figsize=(10, fig_height(len(pss_pd))))
    ax.barh(pss_pd["player_name"], pss_pd["press_setter_share"])

    for i, row in pss_pd.iterrows():
        ax.text(
            row["press_setter_share"] + 0.005,
            i,
            f"{row['press_setter_share']:.0%} ({int(row['press_initiations'])}/{int(row['team_total_chains'])})",
            va="center",
            fontsize=9
        )

    ax.invert_yaxis()
    ax.set_xlabel("press_setter_share %")
    ax.set_ylabel("player_name")
    ax.set_title(f"Press Setter Leadership — {team}")
    ax.set_xlim(0, max(0.05, pss_pd["press_setter_share"].max() * 1.15))
    st.pyplot(fig, clear_figure=True)


def _align_to_ltr_coords_pd(df_xy_pd):
    out = df_xy_pd.copy()
    x = out["x_start"].astype(float)
    y = out["y_start"].astype(float)

    side = out["attacking_side"].astype(str).str.lower()
    attacking_left_to_right = side.isin(["left_to_right", "ltr"]) | side.str.contains("left_to_right")

    out["_x"] = np.where(attacking_left_to_right, x, -x)
    out["_y"] = y
    out["_x"] = out["_x"] + 52.5
    out["_y"] = out["_y"] + 34.0
    return out


def plot_press_setter_heatmaps(obe_pl: pl.DataFrame, pss_team: str, top2: list[str]):
    require_cols(
        obe_pl,
        ["pressing_chain", "index_in_pressing_chain", "team_shortname", "player_name", "x_start", "y_start", "attacking_side"],
        "OBE (Heatmaps)"
    )

    press_inits_pl = obe_pl.filter(
        (pl.col("pressing_chain") == True) &
        (pl.col("index_in_pressing_chain") == 1) &
        (pl.col("team_shortname") == pss_team)
    )

    if press_inits_pl.height == 0 or len(top2) == 0:
        st.info("No press initiations available to plot.")
        return

    press_inits_pd = press_inits_pl.to_pandas()
    press_inits_pd["player_name"] = press_inits_pd["player_name"].astype(str).str.strip()
    press_inits_pd = press_inits_pd.dropna(subset=["x_start", "y_start", "attacking_side"])
    if press_inits_pd.empty:
        st.info("No plot-eligible press initiations (missing x/y/attacking_side).")
        return

    press_inits_pd = _align_to_ltr_coords_pd(press_inits_pd)

    pitch = Pitch(pitch_type="custom", pitch_length=105, pitch_width=68)
    fig, axs = pitch.draw(ncols=2, figsize=(14, 6), constrained_layout=True)
    axs = np.array(axs).ravel()

    bstats = []
    for player in top2:
        pdat = press_inits_pd[press_inits_pd["player_name"] == player]
        if pdat.empty:
            bstats.append(None)
        else:
            bstats.append(pitch.bin_statistic(pdat["_x"], pdat["_y"], statistic="count", bins=(6, 4)))

    vmax = max([bs["statistic"].max() for bs in bstats if bs is not None] + [1])

    for ax, player, bs in zip(axs, top2, bstats):
        if bs is None:
            ax.set_title(f"{player} (no plot-eligible initiations)")
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
            continue

        pitch.heatmap(bs, ax=ax, alpha=0.7, vmin=0, vmax=vmax)
        pitch.label_heatmap(bs, ax=ax, fontsize=11, color="black", ha="center", va="center", exclude_zeros=True)
        ax.set_title(player)

    st.pyplot(fig, clear_figure=True)
    st.caption("Fixed 105×68 pitch. Direction aligned so opponent goal is always on the RIGHT (→). Shared colour scale.")


def plot_press_break_ranking(rank_pl: pl.DataFrame, selected_team: str, top_n_cap: int = 20):
    if rank_pl.height == 0:
        st.info("No press-break failures detected after filters. Try increasing radius or END_WINDOW_S.")
        return

    view_rank = rank_pl.filter(pl.col("defending_team") == selected_team)
    st.markdown("### Ranked Players (Press Break Failures)")
    st.dataframe(view_rank.to_pandas())

    if view_rank.height == 0:
        st.info("No attributed press-break failures for the selected team in this match.")
        return

    top_n = st.slider(
        "Top N (Press Break Failures)",
        1,
        min(top_n_cap, view_rank.height),
        min(15, view_rank.height),
        key="rank_topn_pf"
    )

    plot_rank_pd = view_rank.head(top_n).sort("press_break_failures").to_pandas()
    fig, ax = plt.subplots(figsize=(10, fig_height(len(plot_rank_pd))))
    ax.barh(plot_rank_pd["player_name"], plot_rank_pd["press_break_failures"])
    ax.set_xlabel("press-break failures (attributed)")
    ax.set_title(f"Press Break Failures — {selected_team}")
    st.pyplot(fig, clear_figure=True)


def plot_pressure_clusters_2d(player_pd, team: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    for cname, sub in player_pd.groupby("cluster_name", sort=False):
        ax.scatter(sub["pressure_style"], sub["defensive_score"], label=cname, alpha=0.85)
        for _, r in sub.iterrows():
            ax.text(
                r["pressure_style"],
                r["defensive_score"],
                str(r["player_name"]),
                fontsize=8,
                ha="left",
                va="bottom",
            )

    ax.set_xlabel("Pressure style (pressing_chain / total_engagements)")
    ax.set_ylabel("Defensive score ((force_backward + stop + reduce) / total_engagements)")
    ax.set_title(f"Player Pressure Archetypes — {team}")
    ax.set_xlim(-0.02, 1.02)
    ax.legend()
    st.pyplot(fig, clear_figure=True)
