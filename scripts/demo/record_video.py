"""Record routing demo videos with the Franka arm: a wide view of the whole robot and
every clip, side by side with a close-up of the clip being worked on.

The shared Insert/Pull/Flatten skills, driven by the ground-truth reference
controller with the heuristic stand-in insertion policy (no RL checkpoint is
trained yet), route the cable through the scene's clips in order. Physics frames
are real time (one frame every 1/fps of simulated time). Free-space gripper moves,
which the executor teleports because they cannot touch the cable, are
interpolated for display only: around the robot base in cylindrical coordinates
(as an arm would move), with the fingers opening and closing in 0.3 s.

    CUDA_VISIBLE_DEVICES=3 python scripts/demo/record_video.py --scenes hairpin6_development:0 --out runs/demo/robot
    python scripts/demo/record_video.py --scenes scatter5_development:0,1,2 --no-video --out runs/demo/dry  # physics only
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from cablerouting.cli import _load_state, _make_backend, _policy
from cablerouting.config import ROOT, load_config
from cablerouting.geometry.frames import wrap_angle
from cablerouting.scenes.generator import load_manifest

FPS = 30
WIDE = (1600, 900)
CLOSE = (704, 900)
BAR = 50
FAMILY_LABEL = {
    "primary10": "primary10: ten-clip study family (16-19 cm stages, 5-50 deg turns)",
    "paper5": "paper5: five-clip calibration family",
    "hairpin6": "hairpin6: sharp turns (75-150 deg per stage)",
    "scatter5": "scatter5: far moves off any trail (28-40 cm jumps, S-bends and hooks)",
    "zigzag5": "zigzag5: far and sharp (24-34 cm legs, 60-120 deg turns)",
}


def fit_distance(cam, points: np.ndarray, direction: np.ndarray, target: np.ndarray, margin: float = 0.06) -> float:
    """Smallest camera distance along ``direction`` (unit, target -> eye) that keeps every point in frame."""
    lo, hi = 0.3, 8.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        cam.look(target + mid * direction, target)
        ok = True
        for p in points:
            uv = cam.project(p)
            if uv is None or not (margin * cam.W < uv[0] < (1 - margin) * cam.W
                                  and (margin * cam.H + BAR) < uv[1] < (1 - margin) * cam.H - BAR):
                ok = False
                break
        lo, hi = (lo, mid) if ok else (mid, hi)
    return hi


class Recorder:
    """SimBackend proxy that turns physics steps and gripper moves into composite video frames."""

    def __init__(self, sim, writer=None, free_speed: float = 0.8, open_speed: float = 0.25):
        self.sim, self.writer = sim, writer
        self.free_speed, self.open_speed = free_speed, open_speed
        self.physics_dt = sim.physics_dt
        self.every = max(1, round(1.0 / (FPS * sim.physics_dt)))
        self.n_steps = self.frames = 0
        self.pose = np.array(sim.park_pose, float)
        self.opening = 0.08
        self.clips, self.route, self.done = [], [], []
        self.target: str | None = None
        self.title = self.label = self.close_label = ""
        self.chapters: list[dict] = []
        self.snapshots: dict[str, np.ndarray] = {}
        self.closest = None
        self.video = writer is not None
        self.wide = None
        if self.video:
            self.ensure_cameras()

    def ensure_cameras(self) -> None:
        if self.wide is not None:
            return
        from cablerouting.sim.video_camera import VideoCamera, disable_observation_products

        disable_observation_products(self.sim)
        self.wide = VideoCamera(self.sim, "wide", WIDE, focal_mm=18.0)
        self.close = VideoCamera(self.sim, "close", CLOSE, focal_mm=16.0)
        self.font = ImageFont.load_default(size=28)
        self.small = ImageFont.load_default(size=23)
        self.tag = ImageFont.load_default(size=20)

    # ---------------------------------------------------------------- proxy
    def __getattr__(self, name):
        return getattr(self.sim, name)

    def step(self, n: int = 1, gripper_target=None) -> None:
        for _ in range(n):
            self.sim.step(1, gripper_target=gripper_target)
            if gripper_target is not None:
                self.pose = np.asarray(gripper_target, float)
            self.n_steps += 1
            if self.n_steps % self.every == 0:
                self.capture()

    def set_gripper_pose(self, pose, opening) -> None:
        pose = np.asarray(pose, float)
        if self.video:
            a, b = self.pose, pose
            ra, rb = np.hypot(*a[:2]), np.hypot(*b[:2])
            ta, dt = np.arctan2(a[1], a[0]), float(wrap_angle(np.arctan2(b[1], b[0]) - np.arctan2(a[1], a[0])))
            dyaw = float(wrap_angle(b[3] - a[3]))
            if abs(dyaw) > np.pi / 2:  # the hand is symmetric under a half turn
                dyaw -= np.sign(dyaw) * np.pi
            path = abs(rb - ra) + 0.5 * (ra + rb) * abs(dt) + abs(b[2] - a[2])
            k = int(np.ceil(max(path / self.free_speed, abs(opening - self.opening) / self.open_speed) * FPS))
            for i in range(1, k):
                s = 0.5 - 0.5 * np.cos(np.pi * i / k)  # ease in/out
                r, th = ra + s * (rb - ra), ta + s * dt
                p = np.array([r * np.cos(th), r * np.sin(th), a[2] + s * (b[2] - a[2]), a[3] + s * dyaw])
                self.sim.set_gripper_pose(p, self.opening + s * (opening - self.opening))
                self.capture()
        self.sim.set_gripper_pose(pose, opening)
        self.pose, self.opening = pose, float(opening)
        if self.video:
            self.capture()

    def hold(self, seconds: float) -> None:
        for _ in range(int(seconds * FPS)):
            self.capture()

    def mark(self, label: str) -> None:
        self.label = label
        self.chapters.append({"t": round(self.frames / FPS, 2), "label": label})

    # ---------------------------------------------------------------- cameras
    def frame_wide(self, scene) -> dict:
        """3/4 view from beyond the clips looking back at the robot; everything in frame."""
        xy = np.array([c.pose.xy for c in self.clips])
        centroid = xy.mean(axis=0)
        az = np.arctan2(centroid[1], centroid[0]) + np.radians(28.0)
        el = np.radians(52.0)
        direction = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        head = np.array(scene.initial_head[:2])
        # Clips, the cable head, the pedestal and the arm's working envelope (the elbow
        # stays below ~0.75 m; the ready-posture hand is at 0.31, 0, 0.6).
        pts = [np.array([*p, 0.03]) for p in xy] + [np.array([*head, 0.0]), np.array([0.31, 0.0, 0.62])]
        pts += [np.array([*(0.15 * np.array([np.cos(a), np.sin(a)])), z]) for a in np.linspace(0, 2 * np.pi, 8)
                for z in (0.0, 0.75)]
        pts = np.array(pts)
        target = np.array([*(0.5 * (pts[:, :2].min(0) + pts[:, :2].max(0))), 0.22])
        d = fit_distance(self.wide, pts, direction, target, margin=0.04)
        self.wide.look(target + d * direction, target)
        return {"eye": self.wide.eye.tolist(), "target": target.tolist()}

    def frame_close(self, clip, face: int) -> None:
        """Diagonal view of a clip from the side opposite the hand's working face.

        ``face`` is the side the hand works on (-1 entry: Insert, Flatten; +1 exit:
        Pull). The camera looks from the other side, so the 20 cm-wide Panda hand
        appears behind the clip instead of covering it, and the cable is seen
        coming through the opening towards the camera.
        """
        u = clip.pose.heading
        n = np.array([-u[1], u[0]])
        others = [c.pose.xy for c in self.clips if c.clip_id != clip.clip_id]
        best = None
        for sgn in (1, -1):
            for off in (np.radians(35.0), np.radians(55.0)):
                dxy = -face * np.cos(off) * u + sgn * np.sin(off) * n
                eye_xy = clip.pose.xy + 0.30 * dxy
                # Keep the camera off the arm's side (towards the base) and away from other clips.
                score = float(np.dot(dxy, clip.pose.xy / np.linalg.norm(clip.pose.xy)))
                score += min(min((np.linalg.norm(eye_xy - o) for o in others), default=1.0), 0.2)
                if best is None or score > best[0]:
                    best = (score, eye_xy)
        target = np.array([*(clip.pose.xy + face * 0.02 * u), 0.01])
        self.close.look(np.array([*best[1], 0.15]), target)

    # ---------------------------------------------------------------- frames
    def capture(self) -> None:
        if not self.video:  # still pose the arm, so the log reports IK reachability
            if self.sim.robot is not None:
                self.sim.robot.update(self.sim.gripper_pose(), self.opening, holding=self.sim._attached is not None)
            return
        from cablerouting.sim.video_camera import capture_all

        wide, close = capture_all(self.sim, [self.wide, self.close], rt_subframes=2)
        canvas = Image.new("RGB", (WIDE[0] + CLOSE[0], WIDE[1]))
        canvas.paste(Image.fromarray(wide), (0, 0))
        canvas.paste(Image.fromarray(close), (WIDE[0], 0))
        d = ImageDraw.Draw(canvas)
        self._labels(d, self.wide, 0, self.tag, small=True)
        self._labels(d, self.close, WIDE[0], self.font, small=False)
        d.line([(WIDE[0], 0), (WIDE[0], WIDE[1])], fill=(20, 22, 26), width=4)
        full_w = WIDE[0] + CLOSE[0]
        d.rectangle([0, 0, full_w, BAR], fill=(20, 22, 26))
        d.text((18, BAR // 2), self.title, font=self.font, fill=(240, 240, 240), anchor="lm")
        d.text((full_w - 18, BAR // 2), self.label, font=self.font, fill=(255, 196, 120), anchor="rm")
        d.rectangle([0, WIDE[1] - BAR, full_w, WIDE[1]], fill=(20, 22, 26))
        x = 18
        for cid in self.route:  # route progress chips
            ok, hot = cid in self.done, cid == self.target
            fill = (120, 200, 120) if ok else (255, 196, 120) if hot else (70, 72, 78)
            w = d.textlength(cid, font=self.small) + 16
            d.rounded_rectangle([x, WIDE[1] - BAR + 9, x + w, WIDE[1] - 9], 6, fill=fill)
            d.text((x + w / 2, WIDE[1] - BAR // 2), cid, font=self.small, fill=(20, 20, 20), anchor="mm")
            x += w + 6
        t = self.n_steps * self.physics_dt
        d.text((WIDE[0] - 14, WIDE[1] - BAR // 2), f"sim {t:5.1f} s  real-time", font=self.small,
               fill=(200, 200, 200), anchor="rm")
        d.text((full_w - 14, WIDE[1] - BAR // 2), self.close_label, font=self.small, fill=(200, 200, 200), anchor="rm")
        frame = np.asarray(canvas, dtype=np.uint8)
        self.writer.stdin.write(frame.tobytes())
        self.last_frame = frame
        if self.target is not None and self.label.startswith("Insert"):
            # Keep the frame where the hand is lowest and nearest the target clip.
            g = self.sim.gripper_pose()
            c = next(c for c in self.clips if c.clip_id == self.target)
            d = float(np.linalg.norm(g[:2] - c.pose.xy)) + 2.0 * float(g[2])
            if self.closest is None or d < self.closest[0]:
                self.closest = (d, frame)
        self.frames += 1

    def _labels(self, d, cam, x0: int, font, small: bool) -> None:
        for c in self.clips:
            uv = cam.project(np.array([c.pose.x, c.pose.y, 0.03 if small else 0.028]))
            if uv is None:
                continue
            u, v = uv[0] + x0, uv[1] - (8 if small else 30)
            if not (x0 + 20 < u < x0 + cam.W - 20 and BAR + 24 < v < cam.H - BAR - 4):
                continue
            hot, ok = c.clip_id == self.target, c.clip_id in self.done
            if not small and not hot:
                continue
            fill = (255, 196, 120) if hot else (150, 220, 150) if ok else (245, 245, 245)
            bbox = d.textbbox((u, v), c.clip_id, font=font, anchor="mb")
            d.rectangle([bbox[0] - 4, bbox[1] - 3, bbox[2] + 4, bbox[3] + 3], fill=fill, outline=(30, 30, 30))
            d.text((u, v), c.clip_id, font=font, fill=(20, 20, 20), anchor="mb")

    def snapshot(self, name: str) -> None:
        if self.video and getattr(self, "last_frame", None) is not None:
            self.snapshots[name] = self.last_frame.copy()


def run_scene(cfg, sim, man: str, index: int, out: Path, video: bool, max_decisions: int | None) -> dict:
    from cablerouting.bridge import SimWorld
    from cablerouting.sim.routing_env import RoutingEnv
    from dloroute.agent import HierarchicalAgent
    from dloroute.controllers.reference import ReferenceController
    from dloroute.skills import SkillLibrary

    _, scenes = load_manifest(ROOT / "manifests" / f"{man}.json")
    scene = scenes[index]
    fam = cfg.family(scene.family)
    out.mkdir(parents=True, exist_ok=True)
    writer = None
    if video:
        env_clean = {k: v for k, v in os.environ.items() if k not in ("LD_LIBRARY_PATH", "PYTHONPATH")}
        writer = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{WIDE[0] + CLOSE[0]}x{WIDE[1]}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
             "-preset", "medium", "-crf", "21", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
             str(out / "routing.mp4")], stdin=subprocess.PIPE, env=env_clean)
    rec = getattr(sim, "_recorder", None)
    if rec is None:
        rec = sim._recorder = Recorder(sim, writer)
    rec.writer, rec.video = writer, writer is not None
    if rec.video:
        rec.ensure_cameras()
    rec.n_steps = rec.frames = 0
    rec.chapters, rec.snapshots, rec.done, rec.target = [], {}, [], None
    env = RoutingEnv(rec, scene, fam.view_x, fam.view_y, cfg.motion, cfg.render)
    env.reset(_load_state(scene.scene_id))
    rec.pose, rec.opening = np.array(sim.park_pose, float), 0.08
    if sim.robot is not None:
        sim.robot.max_pos_err = sim.robot.max_rot_err = 0.0
    rec.clips = env.clips
    route = scene.route(len(scene.clips))
    rec.route = route
    by_id = {c.clip_id: c for c in env.clips}
    rec.title = FAMILY_LABEL.get(scene.family, scene.family) + f"  ·  {scene.scene_id}"
    log = {"scene_id": scene.scene_id, "family": scene.family, "route": route,
           "stages": scene.stages, "decisions": []}
    if video:
        log["wide_camera"] = rec.frame_wide(scene)
        rec.frame_close(by_id[route[0]], -1)
        rec.close_label = f"close-up {route[0]}, seen from its exit side"
        rec.mark(f"route {' > '.join(route)}")
        rec.hold(1.5)
        rec.snapshot("start")
    world = SimWorld(env)
    agent = HierarchicalAgent(ReferenceController(world.scene, route, cfg.skills, cfg.reference, render_images=False),
                              SkillLibrary(_policy(cfg, "heuristic"), cfg.skills, cfg.obs, cfg.reward))
    budget = cfg.study.per_clip_budget * len(route)
    max_decisions = max_decisions or budget
    t0 = time.monotonic()
    terminal = "budget"
    for i in range(max_decisions):
        choice = agent.decide(world, i, i, budget)
        d, rule = choice.decision, choice.info["rule"]
        if d.done:
            terminal = "complete"
            break
        rec.target = d.target_clip
        if video:
            face = 1 if d.skill == "Pull" else -1
            rec.frame_close(by_id[d.target_clip], face)
            rec.close_label = f"close-up {d.target_clip}, seen from its {'entry' if face > 0 else 'exit'} side"
        rec.mark(f"{d.skill}({d.target_clip})")
        rec.closest = None
        res = agent.act(d, world)
        after = env.routing(route)
        rec.done = [t.clip_id for t in after.sequence][: after.valid_prefix]
        entry = {"i": i, "decision": d.to_dict(), "rule": rule, "ops": res.physical_ops, "collided": res.collided,
                 "insert_success": res.insert_success, "valid_prefix": after.valid_prefix,
                 "sequence": after.summary()["sequence"], "sim_s": round(rec.n_steps * rec.physics_dt, 2)}
        if res.collided:  # where and how the gripper met a clip
            last = res.details[-1] if res.details else {}
            ex = last.get("exec", last)
            entry["collision"] = {k: ex.get(k) for k in ("label", "collision_clips", "collision_phase", "grasp_xyz")}
            if "action" in last:
                entry["collision"]["action"] = last["action"]
        log["decisions"].append(entry)
        print(f"[video] {scene.scene_id}", entry, flush=True)
        if d.skill == "Insert" and res.insert_success and video and rec.closest is not None:
            rec.snapshots[f"insert_{d.target_clip}"] = rec.closest[1]
        if res.collided:
            terminal = "collision"
            break
    final = env.routing(route)
    rec.target = None
    rec.done = [t.clip_id for t in final.sequence][: final.valid_prefix]
    if video:
        rec.mark(f"threaded {final.valid_prefix}/{len(route)}" + ("  ·  complete" if final.complete else ""))
        rec.hold(2.5)
        rec.snapshot("final")
        writer.stdin.close()
        writer.wait()
        for name, img in rec.snapshots.items():
            Image.fromarray(img).save(out / f"{name}.jpg", quality=88)
    log.update({"terminal": terminal, "complete": bool(final.complete), "valid_prefix": final.valid_prefix,
                "n_clips": len(route), "decisions_used": len(log["decisions"]),
                "physical_ops": int(sum(e["ops"] for e in log["decisions"])),
                "sim_time_s": round(rec.n_steps * rec.physics_dt, 2), "wall_s": round(time.monotonic() - t0, 1),
                "frames": rec.frames, "chapters": rec.chapters, "final_sequence": final.summary()["sequence"]})
    if sim.robot is not None:
        log["ik_max_pos_err_mm"] = 1e3 * sim.robot.max_pos_err
        log["ik_max_rot_err_deg"] = float(np.degrees(sim.robot.max_rot_err))
    (out / "log.json").write_text(json.dumps(log, indent=1, default=str))
    print(f"[video] {scene.scene_id} done: {terminal} {final.valid_prefix}/{len(route)} "
          f"sim {log['sim_time_s']} s wall {log['wall_s']} s", flush=True)
    return log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True, help="manifest:i[,j...][;manifest:k...], e.g. hairpin6_development:0,1")
    ap.add_argument("--out", default="runs/demo/robot")
    ap.add_argument("--no-video", action="store_true", help="physics only (outcome check)")
    ap.add_argument("--max-decisions", type=int, default=None)
    ap.add_argument("--warmup", action="store_true",
                    help="run each scene once without video first; the first episode after loading a scene ends "
                         "~0.1 mm away from later replays, which are identical with or without rendering")
    args = ap.parse_args()
    cfg = load_config()
    sim = _make_backend(cfg)
    summary = []
    for spec in args.scenes.split(";"):
        man, idx = spec.split(":")
        for i in map(int, idx.split(",")):
            _, scenes = load_manifest(ROOT / "manifests" / f"{man}.json")
            if args.warmup:
                w = run_scene(cfg, sim, man, i, Path(args.out) / "warmup" / scenes[i].scene_id, False,
                              args.max_decisions)
                print(f"[video] warmup {w['scene_id']}: {w['terminal']} {w['valid_prefix']}/{w['n_clips']}", flush=True)
            log = run_scene(cfg, sim, man, i, Path(args.out) / scenes[i].scene_id, not args.no_video,
                            args.max_decisions)
            summary.append({k: log.get(k) for k in ("scene_id", "family", "terminal", "complete", "valid_prefix",
                                                    "n_clips", "decisions_used", "physical_ops", "sim_time_s",
                                                    "wall_s", "ik_max_pos_err_mm")})
    print("[video] summary", json.dumps(summary), flush=True)
    sim.close()


if __name__ == "__main__":
    main()
