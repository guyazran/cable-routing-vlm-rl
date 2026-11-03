"""Parallel-jaw gripper collision model (Panda-hand-like dimensions).

Gripper frame: origin at the grasp point (the held cable centreline point),
+x along the held cable tangent pointing towards the head, +y the finger
closing axis, +z up. The gripper yaw therefore equals the heading of the held
cable segment, which is the rotation parameter of the paper's via points.

The simulated gripper is kinematic and moves the cable only through a rigid
grasp attachment; its fingers do not collide with the cable. Only gripper/clip
overlap matters for the collision rule (see geometry.collision).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry.collision import OrientedBox
from ..geometry.frames import rot2


@dataclass(frozen=True)
class GripperDims:
    finger_width: float = 0.020  # along the cable (x)
    finger_thickness: float = 0.008  # along the closing axis (y)
    finger_length: float = 0.054  # vertical
    fingertip_below_grasp: float = 0.004  # fingertips extend this far below the grasp point
    max_opening: float = 0.080
    palm_size: tuple[float, float, float] = (0.06, 0.20, 0.06)  # x, y, z


def gripper_boxes(pose_xyzyaw: np.ndarray, opening: float, dims: GripperDims = GripperDims()) -> list[OrientedBox]:
    """World-frame collision boxes for a gripper pose (x, y, z, yaw) and jaw opening."""
    x, y, z, yaw = pose_xyzyaw
    bottom = z - dims.fingertip_below_grasp
    finger_cz = bottom + dims.finger_length / 2
    off = opening / 2 + dims.finger_thickness / 2
    R = rot2(yaw)
    boxes = []
    for sgn in (1.0, -1.0):
        c = np.array([x, y]) + R @ np.array([0.0, sgn * off])
        boxes.append(OrientedBox(np.array([c[0], c[1], finger_cz]),
                                 np.array([dims.finger_width, dims.finger_thickness, dims.finger_length]), yaw))
    palm_cz = bottom + dims.finger_length + dims.palm_size[2] / 2
    boxes.append(OrientedBox(np.array([x, y, palm_cz]), np.array(dims.palm_size), yaw))
    return boxes
