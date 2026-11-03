"""Exact gripper/clip collision test for yaw-only oriented boxes.

The simulated gripper is kinematic and the clips are static, and PhysX reports no
contacts between those body types. The collision termination rule is therefore
evaluated geometrically at every control step of every primitive: the gripper's
collision boxes are tested against each clip's solid boxes. Cable-clip contact
is ordinary physics and never counts as a collision.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .clip import Clip
from .frames import rot2


@dataclass(frozen=True)
class OrientedBox:
    """Box with centre (x, y, z), size (sx, sy, sz) and yaw about world z."""

    center: np.ndarray
    size: np.ndarray
    yaw: float

    def corners_xy(self) -> np.ndarray:
        hx, hy = self.size[0] / 2, self.size[1] / 2
        local = np.array([[hx, hy], [-hx, hy], [-hx, -hy], [hx, -hy]])
        return local @ rot2(self.yaw).T + self.center[:2]


def _overlap_1d(a_lo, a_hi, b_lo, b_hi, margin) -> bool:
    return a_lo < b_hi + margin and b_lo < a_hi + margin


def boxes_overlap(a: OrientedBox, b: OrientedBox, margin: float = 0.0) -> bool:
    az = (a.center[2] - a.size[2] / 2, a.center[2] + a.size[2] / 2)
    bz = (b.center[2] - b.size[2] / 2, b.center[2] + b.size[2] / 2)
    if not _overlap_1d(*az, *bz, margin):
        return False
    ca, cb = a.corners_xy(), b.corners_xy()
    for yaw in (a.yaw, b.yaw):
        for axis in (np.array([np.cos(yaw), np.sin(yaw)]), np.array([-np.sin(yaw), np.cos(yaw)])):
            pa, pb = ca @ axis, cb @ axis
            if not _overlap_1d(pa.min(), pa.max(), pb.min(), pb.max(), margin):
                return False
    return True


@lru_cache(maxsize=4096)
def clip_world_boxes(clip: Clip) -> tuple[OrientedBox, ...]:
    out = []
    for b in clip.boxes():
        c = clip.pose.to_world(np.array(b.center))
        out.append(OrientedBox(c, np.array(b.size), clip.pose.yaw))
    return tuple(out)


def colliding_clips(gripper_boxes: list[OrientedBox], clips: list[Clip], margin: float = 0.0) -> list[str]:
    hits = []
    g_xy = np.array([g.center[:2] for g in gripper_boxes])
    g_r = max(float(np.linalg.norm(g.size[:2])) for g in gripper_boxes) / 2
    for clip in clips:
        # Cheap bounding-circle rejection before the exact test.
        if np.min(np.linalg.norm(g_xy - clip.pose.xy, axis=1)) > clip.footprint_radius + g_r + margin:
            continue
        cboxes = clip_world_boxes(clip)
        if any(boxes_overlap(g, c, margin) for g in gripper_boxes for c in cboxes):
            hits.append(clip.clip_id)
    return hits
