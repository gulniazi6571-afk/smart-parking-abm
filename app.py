"""
Streamlit dashboard for the Keele University campus parking simulation.

Reads the committed results from data/results/ and displays:
  - KPI cards: search time, utilisation, CO₂ savings, p-value
  - Box plot of search time distributions across 30 replications
  - Headline metrics bar chart
  - Per-replication scatter plot
  - CO₂ and methodology notes

Run with:
  streamlit run dashboard/app.py
"""

import os
import json
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Smart Parking ABM — Keele University",
    page_icon="🅿️",
    layout="wide",
)

RESULTS_DIR = os.path.join("data", "results")
SUMMARY_PATH = os.path.join(RESULTS_DIR, "summary.json")
CSV_PATH     = os.path.join(RESULTS_DIR, "simulation_results.csv")


@st.cache_data
def load_data():
    if not os.path.exists(SUMMARY_PATH) or not os.path.exists(CSV_PATH):
        return None, None
    with open(SUMMARY_PATH) as f:
        summary = json.load(f)
    df = pd.read_csv(CSV_PATH)
    return summary, df


# ─── Title ────────────────────────────────────────────────────────────────────
st.title("🅿️  Intelligent Car Parking ABM — Keele University Campus")
st.caption("COM748 MSc Research Project  ·  Ulster University")
st.divider()

summary, df = load_data()

if summary is None:
    st.warning(
        "No results files found.  "
        "Run `python src/simulation.py` first, then refresh this page."
    )
    st.stop()

fcfs  = summary["fcfs"]
agent = summary["agent"]
ttest = summary["paired_t_test"]

reduction_pct = (
    (fcfs["search_time_mean_s"] - agent["search_time_mean_s"])
    / fcfs["search_time_mean_s"] * 100
)
util_gain = agent["utilisation_mean_pct"] - fcfs["utilisation_mean_pct"]

# ─── KPI Cards ────────────────────────────────────────────────────────────────
st.subheader("Key Performance Indicators")
c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("FCFS Avg Search Time",
          f"{fcfs['search_time_mean_s']:.0f} s")

c2.metric("Agent-Based Avg Search Time",
          f"{agent['search_time_mean_s']:.0f} s",
          delta=f"↓ {reduction_pct:.1f}%",
          delta_color="normal")

c3.metric("Space Utilisation (Agent)",
          f"{agent['utilisation_mean_pct']:.0f}%",
          delta=f"+{util_gain:.0f} pp vs FCFS",
          delta_color="normal")

c4.metric("Est. CO₂ Saving",
          f"≈ {agent['co2_mean']:.0f} kg/day",
          delta=f"≈ {agent['annual_co2_tonnes']} t/year")

sig_label = "✓ p < 0.05" if ttest["significant"] else "✗ p ≥ 0.05"
c5.metric("Paired t-test",
          f"p = {ttest['p_value']:.4f}",
          delta=sig_label,
          delta_color="normal" if ttest["significant"] else "inverse")

st.divider()

# ─── Charts ───────────────────────────────────────────────────────────────────
left, right = st.columns(2)

with left:
    st.subheader("Search Time Distribution (30 replications)")
    fig = go.Figure()
    fig.add_trace(go.Box(y=df["fcfs_search_time_s"], name="FCFS Baseline",
                         marker_color="#EF4444", boxmean="sd"))
    fig.add_trace(go.Box(y=df["agent_search_time_s"], name="Agent-Based",
                         marker_color="#3B82F6", boxmean="sd"))
    fig.update_layout(yaxis_title="Search Time (seconds)", height=380)
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Headline Comparison")
    fig2 = make_subplots(rows=1, cols=2,
                         subplot_titles=("Avg Search Time (s)",
                                         "Space Utilisation (%)"))
    for col_idx, (fv, av) in enumerate([
        (fcfs["search_time_mean_s"], agent["search_time_mean_s"]),
        (fcfs["utilisation_mean_pct"], agent["utilisation_mean_pct"]),
    ], start=1):
        fig2.add_trace(go.Bar(name="FCFS", x=["FCFS"], y=[fv],
                              marker_color="#EF4444", showlegend=(col_idx == 1)),
                       row=1, col=col_idx)
        fig2.add_trace(go.Bar(name="Agent-Based", x=["Agent"], y=[av],
                              marker_color="#3B82F6", showlegend=(col_idx == 1)),
                       row=1, col=col_idx)
    fig2.update_layout(height=380, barmode="group")
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("Per-Replication Search Times")
fig3 = px.scatter(df, x="replication",
                  y=["fcfs_search_time_s", "agent_search_time_s"],
                  color_discrete_map={
                      "fcfs_search_time_s": "#EF4444",
                      "agent_search_time_s": "#3B82F6"},
                  labels={"value": "Search Time (s)", "replication": "Replication"})
fig3.update_layout(legend_title_text="Condition", height=320)
st.plotly_chart(fig3, use_container_width=True)

# ─── Methodology ──────────────────────────────────────────────────────────────
with st.expander("Methodology & CO₂ Estimation"):
    st.markdown(f"""
**CO₂ Estimation (EPA method)**

Saved search distance = (FCFS mean − Agent mean) × cruising speed (15 km/h) × vehicles/day.  
Fuel saved = distance / 25 mpg. CO₂ = fuel × **8.89 kg/gallon** (EPA conversion factor).  
This is a conservative lower bound – idling and tyre/brake particulates excluded.  
*Source: Al-Khafajiy et al. (2020).*

**Statistical Testing**

Paired t-test (`scipy.stats.ttest_rel`), N = {ttest.get('n_replications', 30) if 'n_replications' in summary else 30} replications, α = {ttest['alpha']}.  
Each replication uses a different random seed (ter Hofstede et al. 2023).

**Datasets**

Three Kaggle datasets (PKLot 695 k records, CNRPark+EXT 145 k, Smart Parking 49 k) are
preprocessed to 15-minute intervals and calibrated to ~2,000 Keele campus bays,
08:00–18:00 weekday enforcement hours.
    """)
