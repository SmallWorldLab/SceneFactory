#!/usr/bin/env python3
"""Summarize the 2x2 v7/v8 × dry/wet eval results."""
import json, statistics, pathlib

BASE = pathlib.Path(__file__).parent.parent / "logs/rsl_rl/scene_factory_goal_reaching_eval"

runs = {
    "v7_dry": "2026-05-02_12-24-21_eval_v7_sysid4_weather_model_600_test64_dry",
    "v8_dry": "2026-05-02_12-29-25_eval_v8_sysid4_noweather_model_300_test64_dry",
    "v7_wet": "2026-05-02_12-34-23_eval_v7_sysid4_weather_model_600_test64_hard_sma2mm",
    "v8_wet": "2026-05-02_12-39-13_eval_v8_sysid4_noweather_model_300_test64_hard_sma2mm",
}

metrics = [
    "success_rate", "collision_rate", "lane_forbidden_rate", "crash_rate",
    "mean_min_ttc_s", "near_miss_rate", "mean_max_drac", "high_drac_rate",
]

results = {}
for label, run in runs.items():
    path = BASE / run / "scene_factory_policy_eval_summary.json"
    data = json.loads(path.read_text())
    # top-level dict already contains aggregate metrics
    results[label] = {m: data[m] for m in metrics}

# --- print table ---
col_w = 13
print(f"{'Metric':<26}" + "".join(f"{k:>{col_w}}" for k in runs))
print("-" * (26 + col_w * len(runs)))
for m in metrics:
    row = f"{m:<26}"
    for label in runs:
        row += f"{results[label][m]:>{col_w}.4f}"
    print(row)

# --- print deltas (wet - dry) ---
print()
print("Wet-vs-dry degradation (wet minus dry):")
print(f"{'Metric':<26}{'v7 Δ':>{col_w}}{'v8 Δ':>{col_w}}")
print("-" * (26 + col_w * 2))
for m in metrics:
    dv7 = results["v7_wet"][m] - results["v7_dry"][m]
    dv8 = results["v8_wet"][m] - results["v8_dry"][m]
    print(f"{m:<26}{dv7:>{col_w}.4f}{dv8:>{col_w}.4f}")
