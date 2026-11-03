"""Franka Panda arm for the Isaac backend, posed by IK from the grasp body.

The Isaac Sim Franka asset is referenced under /World/Robot with all of its
physics removed (articulation, bodies, joints, colliders), so it cannot touch the
cable, the clips or the solver. Before every render the hand is placed on the
grasp body: IK puts the point between the fingertips at the grasp point
(``fingertip_below_grasp`` above the fingertips, as in robot.gripper) with a
top-down hand whose closing axis matches the gripper yaw, and the fingers open to
the commanded jaw opening. Link poses are written as USD transforms.

A static box collider at the base footprint keeps the cable out of the pedestal.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..robot.franka import FrankaIK, Q_READY, base_transform, finger_frames, link_frames
from ..robot.gripper import GripperDims
from .isaac_backend import _mat_to_quat_wxyz

ROOT = Path(__file__).resolve().parents[3]
LINKS = [f"panda_link{i}" for i in range(8)] + ["panda_hand", "panda_leftfinger", "panda_rightfinger"]


class FrankaVisual:
    root = "/World/Robot"

    def __init__(self, sim, spec, gripper: GripperDims = GripperDims()):
        from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

        self._Gf = Gf
        st = sim.stage
        path = Path(spec.usd)
        path = path if path.is_absolute() else ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: run scripts/install/fetch_robot_assets.py")
        prim = st.DefinePrim(self.root, "Xform")
        prim.GetReferences().AddReference(str(path))
        vsets = prim.GetVariantSets()
        bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        # Pedestal footprint from the plain base mesh: the detailed variant adds a cable
        # bundle that hangs from the base to the floor, which must not become a collider.
        if vsets.HasVariantSet("Mesh"):
            vsets.GetVariantSet("Mesh").SetVariantSelection("Performance")
        base_rng = bbox.ComputeWorldBound(st.GetPrimAtPath(f"{self.root}/panda_link0")).ComputeAlignedRange()
        self.base_box = (np.array(base_rng.GetMin()), np.array(base_rng.GetMax()))
        bbox.Clear()
        if vsets.HasVariantSet("Mesh"):
            vsets.GetVariantSet("Mesh").SetVariantSelection(spec.mesh_variant)
        if vsets.HasVariantSet("Gripper"):
            vsets.GetVariantSet("Gripper").SetVariantSelection("Default")

        # Strip physics: the arm is visual only.
        apis = [UsdPhysics.ArticulationRootAPI, PhysxSchema.PhysxArticulationAPI, UsdPhysics.RigidBodyAPI,
                PhysxSchema.PhysxRigidBodyAPI, UsdPhysics.MassAPI, UsdPhysics.CollisionAPI,
                UsdPhysics.MeshCollisionAPI, PhysxSchema.PhysxCollisionAPI]
        prims = list(Usd.PrimRange(prim))
        for p in prims:
            if p.IsA(UsdPhysics.Joint):
                p.SetActive(False)
                continue
            for api in apis:
                if p.HasAPI(api):
                    p.RemoveAPI(api)

        # The asset's default pose is q = 0 with closed fingers: check the kinematic model
        # against it link by link before taking over the transforms.
        self.base = base_transform(*spec.base_pose)
        xc = UsdGeom.XformCache()
        fk = link_frames(np.zeros(7), self.base)
        fk.update(finger_frames(fk["panda_hand"], 0.0))
        worst = 0.0
        for name in LINKS:
            T = np.array(xc.GetLocalToWorldTransform(st.GetPrimAtPath(f"{self.root}/{name}"))).T
            T_fk = fk[name]
            worst = max(worst, float(np.abs(T[:3, 3] - T_fk[:3, 3]).max()),
                        0.1 * float(np.abs(T[:3, :3] - T_fk[:3, :3]).max()))
        if worst > 2e-3:
            raise RuntimeError(f"Franka USD does not match the kinematic model (worst {worst:.4f})")
        self.fk_check = worst

        # Fingertip depth below the hand origin (hand z points down in the default pose).
        tip_z = min(bbox.ComputeWorldBound(st.GetPrimAtPath(f"{self.root}/{f}")).ComputeAlignedRange().GetMin()[2]
                    for f in ("panda_leftfinger", "panda_rightfinger"))
        self.fingertip_depth = float(fk["panda_hand"][2, 3] - tip_z)
        self.ik = FrankaIK(self.base, point_depth=self.fingertip_depth - gripper.fingertip_below_grasp)
        if spec.base_collider:
            lo, hi = self.base_box
            lo = np.array([lo[0], lo[1], 0.0])
            cube = UsdGeom.Cube.Define(st, "/World/RobotBaseCollider")
            cube.CreateSizeAttr(1.0)
            cx = UsdGeom.XformCommonAPI(cube)
            cx.SetTranslate(Gf.Vec3d(*map(float, (lo + hi) / 2)))
            cx.SetScale(Gf.Vec3f(*map(float, hi - lo)))
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            UsdGeom.Imageable(cube).MakeInvisible()
            sim._physics_bind(cube.GetPrim())

        self._ops = {}
        for name in LINKS:
            xf = UsdGeom.Xformable(st.GetPrimAtPath(f"{self.root}/{name}"))
            xf.ClearXformOpOrder()
            self._ops[name] = (xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, "fk"),
                               xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble, "fk"))
        # The floating gripper's boxes stay as the (invisible) physics grasp body.
        UsdGeom.Imageable(st.GetPrimAtPath("/World/Gripper")).MakeInvisible()
        self.q = Q_READY.copy()
        self.branch: float | None = None
        self.max_pos_err = 0.0
        self.max_rot_err = 0.0
        self.last = None

    def update(self, pose: np.ndarray, opening: float, holding: bool = False) -> None:
        """Pose the arm so its grasp point and yaw match ``pose`` (x, y, z, yaw)."""
        p, yaw = np.asarray(pose[:3], float), float(pose[3])
        res = self.ik.solve(p, yaw, self.q, keep_branch=self.branch, iters=60)
        wrist = min(res.q[6] + 2.8973, 2.8973 - res.q[6])
        if not holding and (res.pos_err > 1e-3 or res.rot_err > 0.02 or wrist < 0.2):
            res = self.ik.solve(p, yaw, self.q, iters=200)  # free to flip the hand half a turn
        self.q, self.branch, self.last = res.q, res.yaw_branch, res
        self.max_pos_err = max(self.max_pos_err, res.pos_err)
        self.max_rot_err = max(self.max_rot_err, res.rot_err)
        frames = link_frames(self.q, self.base)
        frames.update(finger_frames(frames["panda_hand"], opening))
        Gf = self._Gf
        for name, T in frames.items():
            t, o = self._ops[name]
            t.Set(Gf.Vec3d(*map(float, T[:3, 3])))
            o.Set(Gf.Quatd(*map(float, _mat_to_quat_wxyz(T[:3, :3]))))
