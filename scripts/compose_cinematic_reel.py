"""
compose_cinematic_reel.py
=========================
Post-processing compositor for SceneFactory cinematic shots.

After running run_cinematic_demo.sh, use this script to:
  1. Grid mosaic  — tile N world videos into a single frame (shows scale)
  2. PiP overlay  — main shot + inset corner shot (e.g. chase + overview)
  3. Crossfade    — smooth transition between two clips

Usage examples:

  # Build a 4×4 grid mosaic from per-env eval videos
  python scripts/compose_cinematic_reel.py grid \
      --input-dir logs/rsl_rl/waymo_physx_256_eval/2026-06-03_19-29-10_friction_groups_64/videos \
      --cols 8 --rows 4 \
      --output artifacts/cinematic/grid_mosaic_32.mp4

  # Picture-in-picture: chase cam main + grid overview inset
  python scripts/compose_cinematic_reel.py pip \
      --main    artifacts/cinematic/shot_C_chase/videos/scene_factory_policy_eval_env0.mp4 \
      --inset   artifacts/cinematic/shot_A_grid_reveal/videos/scene_factory_policy_eval_env0.mp4 \
      --pip-corner bottom-right \
      --pip-scale  0.28 \
      --output artifacts/cinematic/chase_with_grid_pip.mp4

  # Crossfade two clips
  python scripts/compose_cinematic_reel.py crossfade \
      --clip-a artifacts/cinematic/shot_A_grid_reveal/videos/scene_factory_policy_eval_env0.mp4 \
      --clip-b artifacts/cinematic/shot_C_chase/videos/scene_factory_policy_eval_env0.mp4 \
      --fade-frames 30 \
      --output artifacts/cinematic/reveal_to_chase.mp4
"""

import argparse
import os
import sys
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError:
    print("ERROR: opencv-python and numpy required.  pip install opencv-python numpy")
    sys.exit(1)


# ─── helpers ─────────────────────────────────────────────────────────────────

def open_video(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    return cap


def video_props(cap: cv2.VideoCapture):
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return w, h, fps, n


def make_writer(path: str, w: int, h: int, fps: float) -> cv2.VideoWriter:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {path}")
    return writer


def read_all_frames(cap: cv2.VideoCapture) -> list[np.ndarray]:
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def resize_frame(frame: np.ndarray, w: int, h: int) -> np.ndarray:
    if frame.shape[1] == w and frame.shape[0] == h:
        return frame
    return cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)


# ─── grid mosaic ─────────────────────────────────────────────────────────────

def cmd_grid(args):
    """Tile multiple per-env eval videos into a single grid frame."""
    input_dir = Path(args.input_dir)
    videos = sorted(input_dir.glob("*.mp4"))
    if not videos:
        print(f"No mp4 files found in {input_dir}")
        sys.exit(1)

    cols = args.cols
    rows = args.rows
    total = cols * rows
    videos = videos[:total]
    print(f"Tiling {len(videos)} videos into {cols}×{rows} grid")

    cell_w = args.cell_width
    cell_h = args.cell_height
    out_w = cols * cell_w
    out_h = rows * cell_h

    # Load all frames for each tile
    all_frames = []
    fps = 30.0
    for v in videos:
        cap = open_video(str(v))
        _, _, fps, _ = video_props(cap)
        frames = read_all_frames(cap)
        all_frames.append(frames)

    max_len = max(len(f) for f in all_frames)
    # Pad shorter clips by looping
    for i in range(len(all_frames)):
        while len(all_frames[i]) < max_len:
            all_frames[i].extend(all_frames[i])
        all_frames[i] = all_frames[i][:max_len]

    # Fill remaining grid slots with black
    blank = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    while len(all_frames) < total:
        all_frames.append([blank] * max_len)

    writer = make_writer(args.output, out_w, out_h, fps)
    print(f"Writing {max_len} frames → {args.output}")
    for fi in range(max_len):
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        for idx in range(total):
            row_i, col_i = divmod(idx, cols)
            frame = resize_frame(all_frames[idx][fi], cell_w, cell_h)
            y0, x0 = row_i * cell_h, col_i * cell_w
            canvas[y0:y0+cell_h, x0:x0+cell_w] = frame
        writer.write(canvas)
        if fi % 30 == 0:
            print(f"  frame {fi}/{max_len}", end="\r", flush=True)
    writer.release()
    print(f"\nGrid mosaic saved → {args.output}")


# ─── picture-in-picture ──────────────────────────────────────────────────────

def cmd_pip(args):
    """Overlay an inset video in a corner of the main video."""
    main_cap  = open_video(args.main)
    inset_cap = open_video(args.inset)

    main_w, main_h, fps, main_n = video_props(main_cap)
    inset_w, inset_h, _, _      = video_props(inset_cap)

    pip_w = int(main_w * args.pip_scale)
    pip_h = int(pip_w * inset_h / inset_w)

    pad = args.pip_padding
    corner = args.pip_corner.lower().replace("-", "_")
    if corner == "bottom_right":
        x0, y0 = main_w - pip_w - pad, main_h - pip_h - pad
    elif corner == "bottom_left":
        x0, y0 = pad, main_h - pip_h - pad
    elif corner == "top_right":
        x0, y0 = main_w - pip_w - pad, pad
    else:  # top_left
        x0, y0 = pad, pad

    # Border thickness around pip
    border = args.pip_border

    writer = make_writer(args.output, main_w, main_h, fps)
    print(f"PiP: {main_w}×{main_h} + {pip_w}×{pip_h} inset @ {corner}")
    print(f"Writing → {args.output}")
    fi = 0
    while True:
        ok_m, frame_m = main_cap.read()
        ok_i, frame_i = inset_cap.read()
        if not ok_m:
            break
        if not ok_i:
            # Loop inset
            inset_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            _, frame_i = inset_cap.read()

        inset_small = resize_frame(frame_i, pip_w, pip_h)

        # Draw border
        if border > 0:
            bx0, by0 = x0 - border, y0 - border
            bx1, by1 = x0 + pip_w + border, y0 + pip_h + border
            bx0, by0 = max(0, bx0), max(0, by0)
            bx1, by1 = min(main_w, bx1), min(main_h, by1)
            frame_m[by0:by1, bx0:bx1] = (255, 255, 255)

        frame_m[y0:y0+pip_h, x0:x0+pip_w] = inset_small

        # Optional label
        if args.pip_label:
            cv2.putText(frame_m, args.pip_label,
                        (x0 + 8, y0 + pip_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 200, 200), 1, cv2.LINE_AA)

        writer.write(frame_m)
        fi += 1
        if fi % 30 == 0:
            print(f"  frame {fi}/{main_n}", end="\r", flush=True)

    main_cap.release()
    inset_cap.release()
    writer.release()
    print(f"\nPiP saved → {args.output}")


# ─── crossfade ────────────────────────────────────────────────────────────────

def cmd_crossfade(args):
    """Crossfade two clips, writing clip_a → fade → clip_b."""
    cap_a = open_video(args.clip_a)
    cap_b = open_video(args.clip_b)

    wa, ha, fps, na = video_props(cap_a)
    wb, hb, _,   nb = video_props(cap_b)

    out_w = max(wa, wb)
    out_h = max(ha, hb)
    fade_n = args.fade_frames
    hold_a = args.hold_a_frames   # frames of clip_a before fade begins
    hold_b = args.hold_b_frames   # frames of clip_b after fade ends

    writer = make_writer(args.output, out_w, out_h, fps)
    print(f"Crossfade: {na} + {nb} frames, {fade_n} fade frames → {args.output}")

    # Phase 1: clip_a (up to hold_a)
    frames_a = read_all_frames(cap_a)
    frames_b = read_all_frames(cap_b)

    used_a = frames_a[:hold_a] if hold_a > 0 else frames_a
    used_b = frames_b[:hold_b] if hold_b > 0 else frames_b

    # Fade source: last fade_n frames of used_a or whole clip if shorter
    fade_src_a = used_a[-fade_n:] if len(used_a) >= fade_n else used_a
    fade_src_b = used_b[:fade_n]  if len(used_b) >= fade_n else used_b
    n_fade = min(len(fade_src_a), len(fade_src_b), fade_n)

    for frame in used_a[:-n_fade]:
        writer.write(resize_frame(frame, out_w, out_h))
    for fi in range(n_fade):
        alpha = fi / max(1, n_fade - 1)
        f_a = resize_frame(fade_src_a[fi], out_w, out_h).astype(np.float32)
        f_b = resize_frame(fade_src_b[fi], out_w, out_h).astype(np.float32)
        blended = (f_a * (1.0 - alpha) + f_b * alpha).astype(np.uint8)
        writer.write(blended)
    for frame in used_b[n_fade:]:
        writer.write(resize_frame(frame, out_w, out_h))

    writer.release()
    print(f"Crossfade saved → {args.output}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="SceneFactory cinematic compositor")
    sub = p.add_subparsers(dest="cmd", required=True)

    # grid
    pg = sub.add_parser("grid", help="Tile per-env eval videos into a grid mosaic")
    pg.add_argument("--input-dir", required=True, help="Directory with per-env .mp4 files")
    pg.add_argument("--cols", type=int, default=8)
    pg.add_argument("--rows", type=int, default=4)
    pg.add_argument("--cell-width",  type=int, default=320, help="Pixels per cell (width)")
    pg.add_argument("--cell-height", type=int, default=180, help="Pixels per cell (height)")
    pg.add_argument("--output", required=True)

    # pip
    pp = sub.add_parser("pip", help="Picture-in-picture overlay")
    pp.add_argument("--main",       required=True, help="Main (background) video")
    pp.add_argument("--inset",      required=True, help="Inset (foreground) video")
    pp.add_argument("--pip-corner", default="bottom-right",
                    choices=["bottom-right","bottom-left","top-right","top-left"])
    pp.add_argument("--pip-scale",   type=float, default=0.28,
                    help="Inset width as fraction of main width")
    pp.add_argument("--pip-padding", type=int,   default=24, help="Corner padding in pixels")
    pp.add_argument("--pip-border",  type=int,   default=3,  help="White border thickness (px)")
    pp.add_argument("--pip-label",   type=str,   default="",
                    help="Optional text label on inset")
    pp.add_argument("--output", required=True)

    # crossfade
    pc = sub.add_parser("crossfade", help="Crossfade two clips")
    pc.add_argument("--clip-a", required=True)
    pc.add_argument("--clip-b", required=True)
    pc.add_argument("--fade-frames",  type=int, default=30)
    pc.add_argument("--hold-a-frames",type=int, default=0,
                    help="Max frames to use from clip_a (0=all)")
    pc.add_argument("--hold-b-frames",type=int, default=0,
                    help="Max frames to use from clip_b (0=all)")
    pc.add_argument("--output", required=True)

    args = p.parse_args()

    if args.cmd == "grid":
        cmd_grid(args)
    elif args.cmd == "pip":
        cmd_pip(args)
    elif args.cmd == "crossfade":
        cmd_crossfade(args)


if __name__ == "__main__":
    main()
