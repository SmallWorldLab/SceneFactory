#!/usr/bin/env python3
"""Generate scene-factory + eval configs for weather/friction sweep.

Creates 4 conditions: dry, light_rain, moderate_rain, heavy_rain
Each condition gets:
  1. A scene_factory YAML with modified friction blocks
  2. An eval YAML pointing to it
"""
import yaml
import copy
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SF_CONFIG_DIR = BASE_DIR / "configs" / "scene_factory" / "generated"

# Template files
SF_TEMPLATE = SF_CONFIG_DIR / "scene_factory_64scene_curated_0326_test.yaml"
EVAL_TEMPLATE = SF_CONFIG_DIR / "eval_fastgoal_v6_patient_model_975_test64_random_od.yaml"

# Weather conditions to sweep
CONDITIONS = {
    "dry": {
        "precip_type": "clear",
        "precip_intensity_mmph": 0.0,
        "water_film_mm": 0.0,
    },
    "light_rain": {
        "precip_type": "rain",
        "precip_intensity_mmph": 4.0,
        "water_film_mm": 0.2,
    },
    "moderate_rain": {
        "precip_type": "rain",
        "precip_intensity_mmph": 10.0,
        "water_film_mm": 0.5,
    },
    "heavy_rain": {
        "precip_type": "shower",
        "precip_intensity_mmph": 16.0,
        "water_film_mm": 0.8,
    },
}


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def save_yaml(data, path):
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, width=120)
    print(f"  wrote {path}")


def main():
    sf_template = load_yaml(SF_TEMPLATE)
    eval_template = load_yaml(EVAL_TEMPLATE)

    for cond_name, cond_params in CONDITIONS.items():
        print(f"\n=== {cond_name} ===")

        # --- 1. Scene-factory config ---
        sf = copy.deepcopy(sf_template)
        for assignment in sf["world"]["assignments"]:
            assignment["friction"]["road_type"] = "AC"
            assignment["friction"]["precip_type"] = cond_params["precip_type"]
            assignment["friction"]["precip_intensity_mmph"] = cond_params["precip_intensity_mmph"]
            assignment["friction"]["water_film_mm"] = cond_params["water_film_mm"]

        sf_filename = f"scene_factory_64scene_curated_0326_test_weather_{cond_name}.yaml"
        sf_path = SF_CONFIG_DIR / sf_filename
        save_yaml(sf, sf_path)

        # --- 2. Eval config ---
        ev = copy.deepcopy(eval_template)
        # Point to modified scene factory config
        ev["scene_factory"]["config_path"] = f"configs/scene_factory/generated/{sf_filename}"
        # Use 10-100m random OD for consistency with paper
        ev["test"]["random_od"] = True
        ev["test"]["random_od_min_travel_m"] = 10.0
        ev["test"]["random_od_max_travel_m"] = 100.0
        # Update run name
        ev["runner"]["run_name"] = f"eval_fastgoal_v6_patient_model_975_test64_weather_{cond_name}"

        eval_filename = f"eval_fastgoal_v6_patient_model_975_test64_weather_{cond_name}.yaml"
        eval_path = SF_CONFIG_DIR / eval_filename
        save_yaml(ev, eval_path)

    print("\nDone! Generated configs for:", list(CONDITIONS.keys()))


if __name__ == "__main__":
    main()
