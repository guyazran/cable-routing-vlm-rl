"""Franka Emika Panda kinematics for the simulated robot arm (numpy only).

The arm is driven kinematically from the gripper pose that the executor commands
(x, y, z, yaw of a top-down parallel-jaw grasp): inverse kinematics gives the
seven joint angles, forward kinematics the link frames to render. Frames follow
Franka's modified (Craig) DH convention, which is also the frame convention of
the Isaac Sim Franka USD (checked against the USD joints when the robot loads).

Hand frame: origin on the flange axis at the hand's mounting face, +z towards
the fingertips, fingers sliding along +-y. For a top-down grasp z points down
and y is the gripper's closing axis, so the hand yaw is fixed up to pi (the hand
is symmetric under a half turn); the solver picks the branch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# (a_{i-1}, d_i, alpha_{i-1}) for joints 1..7, then the flange (fixed).
DH = np.array([
    [0.0, 0.333, 0.0],
    [0.0, 0.0, -np.pi / 2],
    [0.0, 0.316, np.pi / 2],
    [0.0825, 0.0, np.pi / 2],
    [-0.0825, 0.384, -np.pi / 2],
    [0.0, 0.0, np.pi / 2],
    [0.088, 0.0, np.pi / 2],
])
FLANGE_D = 0.107
HAND_YAW = -np.pi / 4  # hand about the flange z axis
FINGER_Z = 0.0584  # finger joint origin along the hand z axis
Q_LOW = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
Q_HIGH = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
Q_READY = np.array([0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4])
# Null-space posture weights: keep the shoulder facing the target and the arm planar
# (q1 = bearing, q3 = q5 = 0); q2/q4/q6 are set by the task, the wrist roll q7 is free.
POSTURE_W = np.array([1.0, 0.05, 1.0, 0.05, 1.0, 0.05, 0.0])


def _tf(a: float, d: float, alpha: float, theta: float) -> np.ndarray:
    ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(theta), np.sin(theta)
    return np.array([[ct, -st, 0.0, a],
                     [st * ca, ct * ca, -sa, -d * sa],
                     [st * sa, ct * sa, ca, d * ca],
                     [0.0, 0.0, 0.0, 1.0]])


def rot_z(theta: float) -> np.ndarray:
    T = np.eye(4)
    c, s = np.cos(theta), np.sin(theta)
    T[:2, :2] = [[c, -s], [s, c]]
    return T


def base_transform(x: float = 0.0, y: float = 0.0, yaw: float = 0.0) -> np.ndarray:
    T = rot_z(yaw)
    T[:2, 3] = x, y
    return T


def link_frames(q: np.ndarray, base: np.ndarray = np.eye(4)) -> dict[str, np.ndarray]:
    """World frames of link0..link7 and the hand for joint angles ``q`` (7,)."""
    frames = {"panda_link0": base.copy()}
    T = base.copy()
    for i in range(7):
        a, d, alpha = DH[i]
        T = T @ _tf(a, d, alpha, q[i])
        frames[f"panda_link{i + 1}"] = T
    flange = T @ _tf(0.0, FLANGE_D, 0.0, 0.0)
    frames["panda_hand"] = flange @ rot_z(HAND_YAW)
    return frames


def finger_frames(hand: np.ndarray, opening: float) -> dict[str, np.ndarray]:
    """Left/right finger frames for a jaw opening (gap between the fingers, m).

    Both finger frames keep the hand's orientation (as in the USD joints); the
    right finger's mesh carries its own half turn.
    """
    d = float(np.clip(opening / 2, 0.0, 0.04))
    out = {}
    for name, s in (("panda_leftfinger", 1.0), ("panda_rightfinger", -1.0)):
        T = hand.copy()
        T[:3, 3] = hand[:3, 3] + hand[:3, :3] @ np.array([0.0, s * d, FINGER_Z])
        out[name] = T
    return out


def top_down_rotation(yaw: float) -> np.ndarray:
    """Hand orientation for a top-down grasp whose closing axis is (-sin yaw, cos yaw, 0)."""
    y = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    z = np.array([0.0, 0.0, -1.0])
    return np.column_stack([np.cross(y, z), y, z])


def _rot_error(R_target: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Rotation vector taking R to R_target (world frame), robust near pi."""
    Rd = R_target @ R.T
    c = np.clip((np.trace(Rd) - 1) / 2, -1.0, 1.0)
    angle = np.arccos(c)
    v = np.array([Rd[2, 1] - Rd[1, 2], Rd[0, 2] - Rd[2, 0], Rd[1, 0] - Rd[0, 1]])
    if angle < 1e-9:
        return 0.5 * v
    if np.pi - angle < 1e-4:
        axis = np.sqrt(np.clip((np.diag(Rd) + 1) / 2, 0.0, None))
        i = int(np.argmax(axis))
        axis = Rd[:, i] + np.eye(3)[i]
        return angle * axis / np.linalg.norm(axis)
    return angle * v / (2 * np.sin(angle))


@dataclass
class IKResult:
    q: np.ndarray
    pos_err: float  # m
    rot_err: float  # rad
    yaw_branch: float  # hand yaw actually solved for (yaw or yaw + pi)


class FrankaIK:
    """Damped-least-squares IK for a point on the hand z axis, top-down orientation.

    ``point_depth`` is the distance along the hand z axis from the hand origin
    to the controlled point (the grasp point between the fingertips).
    """

    def __init__(self, base: np.ndarray = np.eye(4), point_depth: float = 0.1034,
                 rot_weight: float = 0.3, damping: float = 0.02, posture_gain: float = 0.3):
        self.base = base
        self.base_inv = np.linalg.inv(base)
        self.depth = point_depth
        self.w = rot_weight
        self.lam = damping
        self.k0 = posture_gain

    def point_and_frames(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        f = link_frames(q, self.base)
        hand = f["panda_hand"]
        return hand[:3, 3] + self.depth * hand[:3, 2], hand[:3, :3], f

    def jacobian(self, q: np.ndarray, frames: dict, p: np.ndarray) -> np.ndarray:
        J = np.zeros((6, 7))
        for i in range(7):
            T = frames[f"panda_link{i + 1}"]
            z, o = T[:3, 2], T[:3, 3]
            J[:3, i] = np.cross(z, p - o)
            J[3:, i] = z
        return J

    def nominal(self, p_target: np.ndarray, q_prev: np.ndarray) -> np.ndarray:
        """Elbow-up posture facing the target; the wrist roll (q7) is left free."""
        local = self.base_inv @ np.append(p_target, 1.0)
        q = Q_READY.copy()
        q[0] = np.clip(np.arctan2(local[1], local[0]), Q_LOW[0] + 0.1, Q_HIGH[0] - 0.1)
        q[6] = q_prev[6]
        return q

    def _solve_branch(self, p_t: np.ndarray, R_t: np.ndarray, q0: np.ndarray, iters: int,
                      tol: float) -> tuple[np.ndarray, float, float]:
        q = np.clip(q0.astype(float), Q_LOW, Q_HIGH)
        qn = self.nominal(p_t, q0)
        w = np.array([1, 1, 1, self.w, self.w, self.w])
        I6, I7 = np.eye(6), np.eye(7)
        for _ in range(iters):
            p, R, f = self.point_and_frames(q)
            e_p, e_r = p_t - p, _rot_error(R_t, R)
            if np.linalg.norm(e_p) < tol and np.linalg.norm(e_r) < 10 * tol:
                break
            J = w[:, None] * self.jacobian(q, f, p)
            e = w * np.concatenate([e_p, e_r])
            # Damping fades out near the solution (accuracy) and stays near singularities.
            lam2 = self.lam ** 2 * min(1.0, float(np.linalg.norm(e)) / 0.05)
            JJt = J @ J.T
            dq = J.T @ np.linalg.solve(JJt + lam2 * I6, e)
            # Posture in the (undamped) null space, so it does not fight the task.
            N = I7 - J.T @ np.linalg.solve(JJt + 1e-9 * I6, J)
            dq += N @ (self.k0 * POSTURE_W * (qn - q))
            step = np.linalg.norm(dq)
            if step > 0.3:
                dq *= 0.3 / step
            q = np.clip(q + dq, Q_LOW, Q_HIGH)
        p, R, _ = self.point_and_frames(q)
        return q, float(np.linalg.norm(p_t - p)), float(np.linalg.norm(_rot_error(R_t, R)))

    def solve(self, p_target, yaw: float, q0: np.ndarray, *, keep_branch: float | None = None,
              iters: int = 200, tol: float = 1e-5) -> IKResult:
        """Joint angles placing the controlled point at ``p_target`` with a top-down hand.

        ``keep_branch``: hand yaw solved last time; while the hand holds the cable
        the branch closest to it is kept so the wrist never flips mid-motion.
        Otherwise both branches are solved and the more accurate one, then the one
        closer to ``q0``, is used.
        """
        p_t = np.asarray(p_target, dtype=float)
        branches = [yaw, yaw + np.pi]
        if keep_branch is not None:
            d = [abs(np.angle(np.exp(1j * (b - keep_branch)))) for b in branches]
            branches = [branches[int(np.argmin(d))]]
        best = None
        for b in branches:
            q, ep, er = self._solve_branch(p_t, top_down_rotation(b), q0, iters, tol)
            # Prefer accuracy (1 mm ~ 0.01 rad), then a wrist roll away from its limits,
            # then continuity with q0.
            margin = min(q[6] - Q_LOW[6], Q_HIGH[6] - q[6])
            key = (round(ep * 1e3 + er * 1e2, 1), -min(margin, 0.5), float(np.linalg.norm(q - q0)))
            if best is None or key < best[0]:
                best = (key, IKResult(q, ep, er, float(np.angle(np.exp(1j * b)))))
        return best[1]
