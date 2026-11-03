"""Schematic grid of the shape x length dataset: one scene per (shape, n_clips) cell.

No simulator: routes are drawn from the manifests (stage arcs, clips, initial cable).
The test split is held out (horizon-check.md Section 9: inspected evaluation layouts
count as used), so only train and validation can be drawn.

    python scripts/demo/plot_dataset.py --split train --out runs/demo/dataset/shapes_train.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cablerouting.config import ROOT, load_config  # noqa: E402
from cablerouting.scenes.generator import _stage_path, load_manifest  # noqa: E402

ROUTE, START, INK, MUTED, FRAME = "#2a78d6", "#eb6834", "#0b0b0b", "#8a8984", "#d9d8d4"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "validation"], default="train")
    ap.add_argument("--pick", type=int, default=0, help="which scene of each cell (by index within the split)")
    ap.add_argument("--out", default="runs/demo/dataset/shapes_train.png")
    args = ap.parse_args()
    cfg = load_config()
    d = cfg.dataset
    payload, scenes = load_manifest(ROOT / "manifests" / f"{d.name}_{args.split}.json")
    by_cell: dict[tuple[str, int], list] = {}
    for s in scenes:
        by_cell.setdefault((s.family, len(s.clips)), []).append(s)

    fig, axes = plt.subplots(len(d.shapes), len(d.n_clips), squeeze=False,
                             figsize=(1.75 * len(d.n_clips), 1.55 * len(d.shapes) + 0.5))
    th = np.linspace(-np.radians(150), np.radians(150), 120)

    def page(xy):  # world x (away from the robot) points up the page
        xy = np.asarray(xy)
        return -xy[..., 1], xy[..., 0]

    for i, shape in enumerate(d.shapes):
        fam = cfg.family(shape)
        for j, n in enumerate(d.n_clips):
            ax = axes[i][j]
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            s = sorted(by_cell[(shape, n)], key=lambda x: x.scene_id)[args.pick]
            vx, vy = fam.view_x, fam.view_y
            ax.plot(*page([[vx[0], vy[0]], [vx[1], vy[0]], [vx[1], vy[1]], [vx[0], vy[1]], [vx[0], vy[0]]]),
                    color=FRAME, lw=0.6)
            for r in fam.reach_radius:
                ax.plot(*page(np.column_stack([r * np.cos(th), r * np.sin(th)])), color=FRAME, lw=0.6, ls=":")
            ax.plot(*page(s.initial_path), color=MUTED, lw=0.9, ls=(0, (2, 1.5)))
            clips = s.clip_objects()
            for a, b in zip(clips[:-1], clips[1:]):
                ax.plot(*page(_stage_path(a.pose, b.pose, fam)[0]), color=ROUTE, lw=1.6, solid_capstyle="round")
            for k, c in enumerate(clips):
                ax.plot(*page(c.pose.to_world(np.array([[0, -0.024], [0, 0.024]]))),
                        color=START if k == 0 else INK, lw=2.2, solid_capstyle="butt")
            ax.plot(0, 0, marker="^", color=MUTED, ms=4.5)
            ax.set_xlim(-vy[1] - 0.02, -vy[0] + 0.02)
            ax.set_ylim(vx[0] - 0.02, vx[1] + 0.02)
            ax.set_aspect("equal")
            ax.text(0.5, -0.02, s.scene_id, transform=ax.transAxes, ha="center", va="top", fontsize=5.5, color=MUTED)
            if i == 0:
                ax.set_title(f"{n} clips", fontsize=9, color=INK)
            if j == 0:
                ax.text(-0.04, 0.5, shape, transform=ax.transAxes, ha="right", va="center", fontsize=9,
                        color=INK, rotation=90)
    fig.suptitle(f"{d.name} dataset, {args.split} split: one scene per cell. Route in blue, C01 in orange, "
                 "initial cable dashed, robot base ▲, camera view and reach radii in grey",
                 fontsize=8.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0.01, 0, 1, 0.97))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    (out.with_suffix(".json")).write_text(json.dumps(
        {"split": args.split, "pick": args.pick, "manifest_generator": payload["generator_version"],
         "scenes": {f"{k[0]}-n{k[1]:02d}": sorted(v, key=lambda x: x.scene_id)[args.pick].scene_id
                    for k, v in by_cell.items()}}, indent=1))
    print(out)


if __name__ == "__main__":
    main()
