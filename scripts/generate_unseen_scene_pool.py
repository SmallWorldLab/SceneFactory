"""
Generate a scene pool YAML containing only scenes NOT used in the training pool.
Usage:
  python scripts/generate_unseen_scene_pool.py \
    --train_pool configs/scene_factory/generated/scene_factory_256scene_random_0414_train_friction.yaml \
    --scene_dir data/processed/waymo_scenes_json \
    --output configs/scene_factory/generated/eval_unseen_199scenes_dry.yaml
"""
import argparse
import os
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_pool", required=True)
    parser.add_argument("--scene_dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.train_pool) as f:
        train_cfg = yaml.safe_load(f)

    train_scenes = {
        entry["scene_json"]
        for entry in train_cfg["world"]["assignments"]
    }

    all_scenes = sorted(f for f in os.listdir(args.scene_dir) if f.endswith(".json"))
    unseen = sorted(s for s in all_scenes if s not in train_scenes)

    print(f"Total scenes: {len(all_scenes)}, training: {len(train_scenes)}, unseen: {len(unseen)}")

    assignments = [
        {
            "scene_json": scene,
            "friction": {
                "road_type": "AC",
                "precip_type": "clear",
                "precip_intensity_mmph": 0.0,
                "water_film_mm": 0.0,
            },
        }
        for scene in unseen
    ]

    out = {
        "io": {"scene_json_dir": args.scene_dir},
        "world": {
            "world_count": len(unseen),
            "grid_cols": 15,
            "assignment_fill_mode": "random_fill",
            "assignments": assignments,
        },
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        yaml.dump(out, f, default_flow_style=False, sort_keys=False)

    print(f"Written: {args.output}")


if __name__ == "__main__":
    main()
