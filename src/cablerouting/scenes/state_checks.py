"""Predeclared checks of a settled initial state (protocol Section 8).

A saved initial state must match the scene it was generated for:
* no clip is threaded or partially entered (the protocol requires no initially threaded clips);
* the head is still inside C01's insertion region (0.16 m square in front of the entry face);
* the cable lies flat on the table (nothing rests on a clip or the robot pedestal);
* the cable stays outside every clip's footprint;
* settling moved no point of the cable more than ``max_displacement`` from the generated polyline.

The checks depend only on the scene and the state, never on any controller.
"""

from __future__ import annotations

import numpy as np

from ..geometry.threading import evaluate_routing, in_insertion_region
from .generator import Scene, _polyline_distance

LIMITS = {"max_displacement_m": 0.02, "max_height_radii": 3.0, "min_clip_clearance_m": 0.0}


def check_settled(scene: Scene, points: np.ndarray, link_length: float, insertion_region: float) -> dict:
    """``points``: head-first centreline (n_links + 1, 3) of the settled cable."""
    pts = np.asarray(points, dtype=float)
    clips = scene.clip_objects()
    routing = evaluate_routing(clips, pts, [])
    nominal = scene.initial_polyline(link_length)
    displacement = float(np.max(np.linalg.norm(pts[:, :2] - nominal[:, :2], axis=1)))
    height = float(pts[:, 2].max())
    clearance = float(min(_polyline_distance(c.pose.xy, pts[:, :2]) - c.footprint_radius for c in clips))
    checks = {
        "no_threading": not routing.sequence and not any(st.partial or st.anomalies for st in routing.clips.values()),
        "head_in_c01_region": bool(in_insertion_region(clips[0], pts[0], insertion_region)),
        "lies_flat": height <= LIMITS["max_height_radii"] * scene.cable_radius,
        "clear_of_clips": clearance >= LIMITS["min_clip_clearance_m"],
        "settled_near_nominal": displacement <= LIMITS["max_displacement_m"],
    }
    return {"passed": all(checks.values()), "failed": [k for k, v in checks.items() if not v],
            "max_displacement_m": round(displacement, 6), "max_height_m": round(height, 6),
            "min_clip_clearance_m": round(clearance, 6)}
