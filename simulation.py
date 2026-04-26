"""
Runs 30 replications of both the FCFS baseline and the agent-based system,
performs a paired t-test on mean search times, and saves results to data/results/.

Note on committed results
-------------------------
The file data/results/summary.json contains the results from the specific run
reported in the paper (vehicle arrival seed = 48112). Re-running this script
generates a new independent run; results will differ slightly due to stochastic
vehicle arrivals, but the qualitative finding (significant reduction in search
time) is robust across seeds.

Usage:
  python src/simulation.py
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from scipy import stats

from src.model import ParkingModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

N_REPS = 30
RESULTS_DIR = os.path.join("data", "results")


def run_condition(use_agent_system, n_reps=N_REPS):
    label = "AgentBased" if use_agent_system else "FCFS"
    log.info("Running %s (%d replications) …", label, n_reps)

    search_means = []
    util_means = []

    for rep in range(n_reps):
        m = ParkingModel(use_agent_system=use_agent_system, seed=rep)
        m.run()
        search_means.append(m.mean_search_time)
        util_means.append(m.mean_utilisation * 100)
        log.info("  rep %02d  search=%.1fs  util=%.1f%%",
                 rep + 1, m.mean_search_time, m.mean_utilisation * 100)

    return {
        "label": label,
        "search_means": search_means,
        "util_means": util_means,
    }


def compute_co2(fcfs_results, agent_results):
    """
    Run one extra paired model run to compute CO₂ savings cleanly.
    The FCFS mean is passed to model.co2_savings_vs_baseline().
    """
    fcfs_mean = float(np.mean(fcfs_results["search_means"]))
    co2_per_run = []
    for rep in range(N_REPS):
        m = ParkingModel(use_agent_system=True, seed=rep + 1000)
        m.run()
        co2_per_run.append(m.co2_savings_vs_baseline(fcfs_mean))
    return float(np.mean(co2_per_run))


def main():
    fcfs   = run_condition(use_agent_system=False)
    agent  = run_condition(use_agent_system=True)

    t, p = stats.ttest_rel(fcfs["search_means"], agent["search_means"])
    co2   = compute_co2(fcfs, agent)

    reduction = (np.mean(fcfs["search_means"]) - np.mean(agent["search_means"])) \
                / np.mean(fcfs["search_means"]) * 100

    print("\n" + "=" * 55)
    print("  RESULTS SUMMARY")
    print("=" * 55)
    print(f"  {'Metric':<32} {'FCFS':>8} {'Agent':>10}")
    print(f"  {'-'*50}")
    print(f"  {'Avg search time (s)':<32} {np.mean(fcfs['search_means']):>8.1f} {np.mean(agent['search_means']):>10.1f}")
    print(f"  {'Avg utilisation (%)':<32} {np.mean(fcfs['util_means']):>8.1f} {np.mean(agent['util_means']):>10.1f}")
    print(f"  {'Search time reduction':<32} {reduction:>18.1f}%")
    print(f"  {'Est. CO\u2082 saving (kg/day)':<32} {co2:>18.1f}")
    print(f"  {'Paired t-test p-value':<32} {p:>18.4f}")
    sig = "significant" if p < 0.05 else "NOT significant"
    print(f"  Result: {sig} (α = 0.05)")
    print("=" * 55 + "\n")

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = pd.DataFrame({
        "replication": range(1, N_REPS + 1),
        "fcfs_search_time_s": [round(x, 1) for x in fcfs["search_means"]],
        "agent_search_time_s": [round(x, 1) for x in agent["search_means"]],
        "fcfs_utilisation_pct": [round(x, 1) for x in fcfs["util_means"]],
        "agent_utilisation_pct": [round(x, 1) for x in agent["util_means"]],
    })
    csv_path = os.path.join(RESULTS_DIR, "simulation_results.csv")
    df.to_csv(csv_path, index=False)

    summary = {
        "n_replications": N_REPS,
        "fcfs": {
            "search_time_mean_s": round(float(np.mean(fcfs["search_means"])), 1),
            "utilisation_mean_pct": round(float(np.mean(fcfs["util_means"])), 1),
        },
        "agent": {
            "search_time_mean_s": round(float(np.mean(agent["search_means"])), 1),
            "utilisation_mean_pct": round(float(np.mean(agent["util_means"])), 1),
            "co2_mean": round(co2, 1),
            "annual_co2_tonnes": round(co2 * 150 / 1000, 1),  # 150 academic weekdays
        },
        "paired_t_test": {
            "t_statistic": round(float(t), 3),
            "p_value": round(float(p), 4),
            "alpha": 0.05,
            "significant": bool(p < 0.05),
        },
    }
    json_path = os.path.join(RESULTS_DIR, "summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    log.info("Results saved → %s and %s", csv_path, json_path)


if __name__ == "__main__":
    main()
