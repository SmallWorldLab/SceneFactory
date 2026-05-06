"""
convert_waymo_tfrecord_to_json.py

Converts Waymo Open Motion Dataset TFRecord files into per-scene JSON files
that SceneFactory can load directly.

Requirements:
    pip install tensorflow waymo-open-dataset-tf-2-12-0 numpy

Usage:
    python scripts/convert_waymo_tfrecord_to_json.py \
        --tfrecord-dir data/waymo_tfrecords \
        --output-dir data/processed/waymo_scenes_json

Each TFRecord file may contain multiple scenes; one JSON is written per scene,
named scene_000000.json, scene_000001.json, etc.
"""

import argparse
import json
import os
import numpy as np


# ---------------------------------------------------------------------------
# Feature schema (Waymo Open Motion Dataset)
# ---------------------------------------------------------------------------

NUM_MAP_SAMPLES = 30000

FEATURES_DESCRIPTION = {}

FEATURES_DESCRIPTION.update({
    'roadgraph_samples/dir':   __import__('tensorflow').io.FixedLenFeature([NUM_MAP_SAMPLES, 3], __import__('tensorflow').float32, default_value=None),
    'roadgraph_samples/id':    __import__('tensorflow').io.FixedLenFeature([NUM_MAP_SAMPLES, 1], __import__('tensorflow').int64,   default_value=None),
    'roadgraph_samples/type':  __import__('tensorflow').io.FixedLenFeature([NUM_MAP_SAMPLES, 1], __import__('tensorflow').int64,   default_value=None),
    'roadgraph_samples/valid': __import__('tensorflow').io.FixedLenFeature([NUM_MAP_SAMPLES, 1], __import__('tensorflow').int64,   default_value=None),
    'roadgraph_samples/xyz':   __import__('tensorflow').io.FixedLenFeature([NUM_MAP_SAMPLES, 3], __import__('tensorflow').float32, default_value=None),
})

FEATURES_DESCRIPTION.update({
    'state/id':                    __import__('tensorflow').io.FixedLenFeature([128],     __import__('tensorflow').float32, default_value=None),
    'state/type':                  __import__('tensorflow').io.FixedLenFeature([128],     __import__('tensorflow').float32, default_value=None),
    'state/is_sdc':                __import__('tensorflow').io.FixedLenFeature([128],     __import__('tensorflow').int64,   default_value=None),
    'state/current/bbox_yaw':      __import__('tensorflow').io.FixedLenFeature([128, 1],  __import__('tensorflow').float32, default_value=None),
    'state/current/valid':         __import__('tensorflow').io.FixedLenFeature([128, 1],  __import__('tensorflow').int64,   default_value=None),
    'state/current/x':             __import__('tensorflow').io.FixedLenFeature([128, 1],  __import__('tensorflow').float32, default_value=None),
    'state/current/y':             __import__('tensorflow').io.FixedLenFeature([128, 1],  __import__('tensorflow').float32, default_value=None),
    'state/current/z':             __import__('tensorflow').io.FixedLenFeature([128, 1],  __import__('tensorflow').float32, default_value=None),
    'state/future/valid':          __import__('tensorflow').io.FixedLenFeature([128, 80], __import__('tensorflow').int64,   default_value=None),
    'state/future/bbox_yaw':       __import__('tensorflow').io.FixedLenFeature([128, 80], __import__('tensorflow').float32, default_value=None),
    'state/future/x':              __import__('tensorflow').io.FixedLenFeature([128, 80], __import__('tensorflow').float32, default_value=None),
    'state/future/y':              __import__('tensorflow').io.FixedLenFeature([128, 80], __import__('tensorflow').float32, default_value=None),
    'state/future/z':              __import__('tensorflow').io.FixedLenFeature([128, 80], __import__('tensorflow').float32, default_value=None),
})


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _to_1d(x):
    return np.asarray(x).reshape(-1)


def _order_polyline_pca_xy(xyz: np.ndarray) -> np.ndarray:
    """Order points along the dominant XY axis using PCA."""
    n = xyz.shape[0]
    if n <= 2:
        return np.arange(n)
    p = xyz[:, :2].astype(np.float64)
    p -= p.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(p, full_matrices=False)
    return np.argsort(p @ vt[0])


def extract_road_polylines(parsed_ex):
    rg_valid = _to_1d(parsed_ex["roadgraph_samples/valid"].numpy()).astype(bool)
    rg_xyz   = parsed_ex["roadgraph_samples/xyz"].numpy()
    rg_dir   = parsed_ex["roadgraph_samples/dir"].numpy()
    rg_id    = _to_1d(parsed_ex["roadgraph_samples/id"].numpy())
    rg_type  = _to_1d(parsed_ex["roadgraph_samples/type"].numpy())

    xyz   = rg_xyz[rg_valid]
    direc = rg_dir[rg_valid]
    ids   = rg_id[rg_valid].astype(np.int64)
    types = rg_type[rg_valid].astype(np.int64)

    keys = np.stack([types, ids], axis=1)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)

    polylines = []
    for gi, (t, i) in enumerate(uniq):
        m   = (inv == gi)
        pts = xyz[m]
        d   = direc[m]
        idx = _order_polyline_pca_xy(pts)
        pts = pts[idx]
        d   = d[idx]
        polylines.append({
            "type": int(t),
            "id":   int(i),
            "n":    int(pts.shape[0]),
            "xyz":  pts.tolist(),
            "dir":  d.tolist(),
        })

    stats = {
        "road_valid_points":    int(xyz.shape[0]),
        "num_groups_total":     int(uniq.shape[0]),
        "num_polylines_saved":  int(len(polylines)),
    }
    return polylines, stats


def extract_agents(parsed_ex):
    cur_valid = _to_1d(parsed_ex["state/current/valid"].numpy()).astype(bool)
    is_sdc    = _to_1d(parsed_ex["state/is_sdc"].numpy()).astype(bool)
    a_type    = _to_1d(parsed_ex["state/type"].numpy())
    a_id      = _to_1d(parsed_ex["state/id"].numpy())

    cx   = _to_1d(parsed_ex["state/current/x"].numpy())
    cy   = _to_1d(parsed_ex["state/current/y"].numpy())
    cz   = _to_1d(parsed_ex["state/current/z"].numpy())
    cyaw = _to_1d(parsed_ex["state/current/bbox_yaw"].numpy())

    f_valid = parsed_ex["state/future/valid"].numpy().astype(bool)
    fx      = parsed_ex["state/future/x"].numpy()
    fy      = parsed_ex["state/future/y"].numpy()
    fz      = parsed_ex["state/future/z"].numpy()
    fyaw    = parsed_ex["state/future/bbox_yaw"].numpy()

    agents = []
    for i in np.where(cur_valid)[0]:
        start = {"x": float(cx[i]), "y": float(cy[i]),
                 "z": float(cz[i]), "yaw": float(cyaw[i])}
        end = None
        js = np.where(f_valid[i])[0]
        if js.size > 0:
            j = int(js[-1])
            end = {"x": float(fx[i, j]), "y": float(fy[i, j]),
                   "z": float(fz[i, j]), "yaw": float(fyaw[i, j]),
                   "t_idx": int(j)}
        agents.append({
            "track_idx":  int(i),
            "is_sdc":     bool(is_sdc[i]),
            "agent_type": int(a_type[i]) if np.isfinite(a_type[i]) else None,
            "agent_id":   float(a_id[i]) if np.isfinite(a_id[i]) else None,
            "start":      start,
            "end":        end,
        })

    sdc_list = [a for a in agents if a["is_sdc"]]
    sdc = sdc_list[0] if sdc_list else None
    items = [a for a in agents if not a["is_sdc"]]
    return items, sdc, len(agents)


# ---------------------------------------------------------------------------
# Main export loop
# ---------------------------------------------------------------------------

def export(tfrecord_dir: str, output_dir: str):
    import tensorflow as tf

    os.makedirs(output_dir, exist_ok=True)

    tfrecord_files = sorted(
        f for f in os.listdir(tfrecord_dir)
        if not f.startswith(".") and os.path.isfile(os.path.join(tfrecord_dir, f))
    )
    if not tfrecord_files:
        raise RuntimeError(f"No TFRecord files found in {tfrecord_dir}")

    print(f"Found {len(tfrecord_files)} TFRecord file(s) in {tfrecord_dir}")

    scene_idx = 0
    for fname in tfrecord_files:
        path = os.path.join(tfrecord_dir, fname)
        ds = tf.data.TFRecordDataset(path, compression_type="")
        n_in_file = 0
        for raw in ds:
            ex = tf.io.parse_single_example(raw, FEATURES_DESCRIPTION)
            polylines, road_stats = extract_road_polylines(ex)
            items, sdc, count_valid = extract_agents(ex)

            payload = {
                "meta": {
                    "source_file": fname,
                    "scene_index_global": scene_idx,
                },
                "road": {
                    "stats": road_stats,
                    "polylines": polylines,
                },
                "agents": {
                    "count_valid": count_valid,
                    "sdc": sdc,
                    "items": items,
                },
            }

            out_path = os.path.join(output_dir, f"scene_{scene_idx:06d}.json")
            with open(out_path, "w") as f:
                json.dump(payload, f)

            scene_idx += 1
            n_in_file += 1

        print(f"  {fname}: {n_in_file} scene(s)")

    print(f"\nDone. Total scenes written: {scene_idx}")
    print(f"Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Waymo Open Motion Dataset TFRecords to SceneFactory scene JSONs."
    )
    parser.add_argument(
        "--tfrecord-dir",
        default="data/waymo_tfrecords",
        help="Directory containing *.tfrecord files (default: data/waymo_tfrecords)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/waymo_scenes_json",
        help="Directory to write scene_XXXXXX.json files (default: data/processed/waymo_scenes_json)",
    )
    args = parser.parse_args()
    export(args.tfrecord_dir, args.output_dir)


if __name__ == "__main__":
    main()
