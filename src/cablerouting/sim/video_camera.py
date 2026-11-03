"""Free perspective cameras for demo videos (not used by any observation).

Each camera is a USD camera with its own Replicator render product. One
orchestrator step renders every enabled product, so several views of the same
instant are captured together with :func:`capture_all`.
"""

from __future__ import annotations

import numpy as np

H_APERTURE_MM = 20.955


def look_at_quat_wxyz(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """USD camera orientation (looks along -Z, +Y up) for a camera at ``eye`` facing ``target``."""
    from .isaac_backend import _mat_to_quat_wxyz

    f = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(f, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, f)
    return _mat_to_quat_wxyz(np.column_stack([right, up, -f]))


class VideoCamera:
    def __init__(self, sim, name: str, resolution: tuple[int, int], focal_mm: float = 18.0):
        import omni.replicator.core as rep
        from pxr import Gf, UsdGeom

        self.sim, self.name = sim, name
        self.W, self.H = resolution
        self.focal_mm = focal_mm
        path = f"/World/Cameras/video_{name}"
        cam = UsdGeom.Camera.Define(sim.stage, path)
        cam.CreateFocalLengthAttr(focal_mm)
        cam.CreateHorizontalApertureAttr(H_APERTURE_MM)
        cam.CreateVerticalApertureAttr(H_APERTURE_MM * self.H / self.W)
        cam.CreateClippingRangeAttr((0.01, 50.0))
        xf = UsdGeom.Xformable(cam)
        self._t, self._q = xf.AddTranslateOp(), xf.AddOrientOp()
        self._Gf = Gf
        self._rep = rep
        self.rp = rep.create.render_product(path, (self.W, self.H))
        self.ann = rep.AnnotatorRegistry.get_annotator("rgb")
        self.ann.attach([self.rp])
        self.eye = self.target = None
        self.look(np.array([1.5, -1.2, 1.2]), np.zeros(3))

    def look(self, eye, target) -> None:
        eye, target = np.asarray(eye, float), np.asarray(target, float)
        self._t.Set(self._Gf.Vec3d(*map(float, eye)))
        self._q.Set(self._Gf.Quatf(*map(float, look_at_quat_wxyz(eye, target))))
        f = (target - eye) / np.linalg.norm(target - eye)
        right = np.cross(f, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        self._basis = (eye, right, np.cross(right, f), f)
        self.eye, self.target = eye, target

    def project(self, p):
        """World point -> pixel (u, v), or None if behind the camera."""
        eye, right, up, f = self._basis
        d = np.asarray(p, float) - eye
        z = float(d @ f)
        if z <= 0.05:
            return None
        fpx = self.focal_mm / H_APERTURE_MM * self.W
        return self.W / 2 + fpx * float(d @ right) / z, self.H / 2 - fpx * float(d @ up) / z

    def read(self) -> np.ndarray:
        return np.asarray(self.ann.get_data())[..., :3].copy()

    def capture(self, rt_subframes: int = 2) -> np.ndarray:
        return capture_all(self.sim, [self], rt_subframes)[0]


def capture_all(sim, cams: list[VideoCamera], rt_subframes: int = 2) -> list[np.ndarray]:
    sim._sync_render()
    cams[0]._rep.orchestrator.step(rt_subframes=rt_subframes, pause_timeline=True, delta_time=0.0)
    return [c.read() for c in cams]


def disable_observation_products(sim) -> None:
    """Stop rendering the planner/crop cameras (speeds up recording; env.render() unusable)."""
    for _, rp, _, _ in sim._cams.values():
        try:
            rp.hydra_texture.set_updates_enabled(False)
        except Exception:  # noqa: BLE001 - older Replicator: all products keep rendering
            pass
