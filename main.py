import streamlit as st
import polars as pl

from src.pressure_profile_components import (
    require_cols,
    build_pressure_profile_pl,
    build_press_breaks_pl,
    build_pressure_clusters_2d_pl,
    plot_engagement_split,
    plot_press_outcomes,
    plot_press_setter_share,
    plot_press_setter_heatmaps,
    plot_press_break_ranking,
    plot_pressure_clusters_2d,
)

# =========================================================
# CONFIG
# =========================================================
GAMES = [
    1886347, 1899585, 1925299, 1953632, 1996435,
    2006229, 2011166, 2013725, 2015213, 2017461
]

# Press-break parameters (event-only)
FPS = 25
CHAIN_GAP_S = 4.0
ELIGIBLE_RADIUS_M = 12.0
END_WINDOW_S = 1.0
TOP_N_FAIL = 20

GITHUB_BASE = "https://raw.githubusercontent.com/SkillCorner/opendata/master/data/matches"


# =========================================================
# DATA LOADING
# =========================================================
@st.cache_data
def load_match(gid: int) -> pl.DataFrame:
    url = f"{GITHUB_BASE}/{gid}/{gid}_dynamic_events.csv"
    return pl.read_csv(url)


@st.cache_data
def build_match_labels(game_ids):
    labels = {}
    for gid in game_ids:
        df0 = load_match(gid)
        if "team_shortname" in df0.columns:
            teams = (
                df0.select(pl.col("team_shortname").drop_nulls().unique().sort())
                .to_series()
                .to_list()
            )
        else:
            teams = []
        matchup = (
            f"{teams[0]} vs {teams[1]}"
            if len(teams) >= 2
            else (teams[0] if len(teams) == 1 else "Unknown teams")
        )
        labels[gid] = f"{matchup} (Match {gid})"
    return labels


# =========================================================
# UI
# =========================================================
st.title("PRESSURE POINT")
st.caption("Deep defensive analytics for pressing, pressure, chains and defensive influence.")

with st.sidebar:
    st.header("Selection")

    match_labels = build_match_labels(GAMES)
    game = st.selectbox(
        "Match",
        GAMES,
        format_func=lambda gid: match_labels.get(gid, str(gid)),
        key="match_select",
    )

    df_pl = load_match(game)

    # OBE slice in POLARS
    require_cols(df_pl, ["event_type"], "Dynamic Events")
    obe_pl = df_pl.filter(pl.col("event_type") == "on_ball_engagement")

    # Build shared tables once
    profile_pl = build_pressure_profile_pl(obe_pl)

    if profile_pl.height == 0:
        st.error("No pressure data available in this match.")
        st.stop()

    teams = profile_pl.select(pl.col("team").drop_nulls().unique().sort()).to_series().to_list()
    if not teams:
        st.error("No teams found in this match.")
        st.stop()

    selected_team = st.selectbox("Team", teams, key="team_filter_main")
    profile_main_pl = profile_pl.filter(pl.col("team") == selected_team)

    st.divider()
    st.subheader("Display controls")

    top_n_players = st.slider(
        "Top N players (charts)",
        1,
        min(50, profile_main_pl.height),
        min(20, profile_main_pl.height),
        key="topn_profiles_sidebar"
    )

    min_eng = st.slider(
        "Minimum engagements (clustering)",
        0, 30, 5,
        key="cluster2d_min_eng_sidebar"
    )

st.markdown(f"### {selected_team}")
st.caption("All views use the same selected match + team. Charts are ordered within each section as noted.")


# =========================================================
# KPI strip (computed in polars)
# =========================================================
if profile_main_pl.height == 0:
    st.info("No pressure data available for the selected team.")
else:
    kpi = profile_main_pl.select(
        pl.sum("total_engagements").alias("team_total_eng"),
        pl.sum("pressing_chain").alias("team_chain_eng"),
        pl.sum("press_initiations").alias("team_press_inits"),
        (pl.sum("force_backward") + pl.sum("stop_danger") + pl.sum("reduce_danger")).alias("team_outcomes"),
    ).to_dicts()[0]

    team_total_eng = int(kpi["team_total_eng"] or 0)
    team_chain_eng = int(kpi["team_chain_eng"] or 0)
    team_press_inits = int(kpi["team_press_inits"] or 0)
    team_outcomes = int(kpi["team_outcomes"] or 0)

    press_leader = (
        profile_main_pl.sort(["press_setter_share", "press_initiations"], descending=[True, True])
        .head(1)
        .to_dicts()[0]
    )
    leader_name = str(press_leader["player_name"])
    leader_share = float(press_leader["press_setter_share"] or 0.0)
    leader_inits = int(press_leader["press_initiations"] or 0)

    chain_reliance = (team_chain_eng / team_total_eng) if team_total_eng > 0 else 0.0
    defensive_impact_rate = (team_outcomes / team_total_eng) if team_total_eng > 0 else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total engagements", f"{team_total_eng}")
    c2.metric("Press initiations", f"{team_press_inits}")
    c3.metric("Chain reliance", f"{chain_reliance:.0%}")
    c4.metric("Defensive impact rate", f"{defensive_impact_rate:.0%}")

    c5 = st.columns(1)[0]
    c5.metric("Press leader", leader_name, f"{leader_share:.0%} ({leader_inits} chains)")

st.divider()


# =========================================================
# Tabs
# =========================================================
tab_profiles, tab_setters, tab_breaks, tab_clusters = st.tabs(
    ["Player Profiles", "Press Setter Share", "Press Breaks", "Archetypes"]
)

# TAB 1
with tab_profiles:
    st.header("Player Pressure Profiles")

    with st.expander("Show table (player pressure profiles)", expanded=False):
        st.dataframe(profile_main_pl.to_pandas(), use_container_width=True)

    plot_profile_pl = profile_main_pl.sort("total_engagements", descending=True).head(top_n_players)
    plot_profile_pd = plot_profile_pl.to_pandas()

    st.subheader("Engagement split + press initiations")
    plot_engagement_split(plot_profile_pd, selected_team)

    st.subheader("Press outcomes")
    plot_press_outcomes(plot_profile_pd, selected_team)

# TAB 2
with tab_setters:
    st.header("Press Setter Share")

    pss_pl = profile_main_pl.sort("press_setter_share", descending=True)
    pss_pd = pss_pl.to_pandas()

    st.subheader("Press setter leadership")
    plot_press_setter_share(pss_pd.reset_index(drop=True), selected_team)

    st.subheader("Press setter heatmaps (Top 2)")
    top2 = (
        pss_pl.sort(["press_setter_share", "press_initiations"], descending=[True, True])
        .head(2)
        .select(pl.col("player_name").cast(pl.Utf8).str.strip_chars())
        .to_series()
        .to_list()
    )
    plot_press_setter_heatmaps(obe_pl, selected_team, top2)

# TAB 3
with tab_breaks:
    st.header("Press Break Responsibility (Failures Only)")

    press_breaks_pl, rank_pl, meta = build_press_breaks_pl(
        df=df_pl,
        fps=FPS,
        chain_gap_s=CHAIN_GAP_S,
        eligible_radius_m=ELIGIBLE_RADIUS_M,
        end_window_s=END_WINDOW_S,
    )

    if "error" in meta:
        st.warning(f"Press Breaks skipped: {meta['error']}")
    else:
        plot_press_break_ranking(rank_pl, selected_team, top_n_cap=TOP_N_FAIL)

        with st.expander("Show attribution filters used", expanded=False):
            st.write(
                f"- Distance ≤ {ELIGIBLE_RADIUS_M}m via `{meta.get('dist_col')}`\n"
                f"- Goal-side: {'ON' if meta.get('goalside_col') else 'OFF (missing col)'}\n"
                f"- Trajectory: {'ON' if meta.get('traj_col') else 'OFF (missing col)'}\n"
                f"- Overload: {'ON' if meta.get('overload_col') else 'OFF (missing col)'}\n"
                f"- Attribution: one player per broken chain (closest defender after filters)"
            )

# TAB 4
with tab_clusters:
    st.header("Player Pressure Archetypes")

    player_pl, cluster_avgs_pl = build_pressure_clusters_2d_pl(
        profile_main=profile_main_pl,
        k=3,
        min_engagements=min_eng,
    )

    if player_pl.height == 0:
        st.info("Not enough eligible players for archetype clustering.")
    else:
        st.subheader("Scatter + archetypes")
        plot_pressure_clusters_2d(player_pl.to_pandas(), selected_team)

        colA, colB = st.columns(2)
        with colA:
            with st.expander("Show player → archetype table", expanded=False):
                st.dataframe(player_pl.to_pandas(), use_container_width=True)
        with colB:
            with st.expander("Show cluster characteristics (averages)", expanded=False):
                st.dataframe(cluster_avgs_pl.to_pandas(), use_container_width=True)
