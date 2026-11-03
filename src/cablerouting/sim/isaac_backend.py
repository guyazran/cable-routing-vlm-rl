"""Isaac Sim 6.2 (PhysX 5, Kit 110.4) backend.

Scene content is authored with plain USD / UsdPhysics / PhysxSchema; runtime
state goes through the PhysX tensor API; stepping uses SimulationManager and
images use Replicator render products refreshed by RenderingManager.render(),
which updates the app without stepping physics.

Cable: a chain of rigid capsules joined by D6 joints (linear DOFs and twist
locked, two bending DOFs with cone limits and damped angular drives), the PhysX
rigid-body rope pattern. Capsule frames point their local +x towards the head.
Gripper: a kinematic rigid body without colliders; grasping creates a fixed
joint to one cable link at the current relative pose (no teleporting) and
releasing deletes it.

Robot: a Franka Emika Panda (Isaac Sim's asset) mounted at the virtual robot
base. The arm is posed by inverse kinematics from the gripper body's pose before
every render, so its hand and fingers sit exactly where the grasp body is; it has
no physics of its own (bodies, joints and colliders are stripped) except a static
collider at the base, which keeps the cable out of the pedestal. Observation
cameras (planner full view and crops) hide the robot, as a wrist-mounted camera
at the observation pose would not see it.

Import this module only after constructing :class:`IsaacBackend`'s SimulationApp
(the constructor does that), because pxr/omni modules need the running app.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

import numpy as np

from ..geometry.cable import link_points
from ..geometry.clip import Clip
from ..geometry.frames import yaw_to_quat_wxyz
from .camera_model import TopDownCamera


@dataclass(frozen=True)
class CableSpec:
    length: float = 2.4
    radius: float = 0.005
    link_length: float = 0.0125
    mass_per_m: float = 0.05
    friction: float = 0.5  # paper: friction coefficient 0.5 (applied to every contact pair)
    bend_limit_deg: float = 60.0  # per joint
    bend_stiffness: float = 2e-5  # angular drive, N*m/deg
    bend_damping: float = 2e-5  # angular drive, N*m*s/deg
    linear_damping: float = 0.2
    angular_damping: float = 0.5
    contact_offset: float = 0.002
    rest_offset: float = 0.0

    @property
    def n_links(self) -> int:
        return int(round(self.length / self.link_length))


@dataclass(frozen=True)
class PhysicsSpec:
    dt: float = 1.0 / 240.0
    solver: str = "TGS"
    position_iterations: int = 32
    velocity_iterations: int = 4
    max_depenetration_velocity: float = 0.5
    gravity: float = 9.81
    enhanced_determinism: bool = True


@dataclass(frozen=True)
class RobotSpec:
    model: str = "franka_panda"  # "none": floating gripper visuals only
    usd: str = "assets/robots/franka_panda/franka.usd"  # scripts/install/fetch_robot_assets.py
    mesh_variant: str = "Quality"
    base_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)  # x, y, yaw of the virtual base
    base_collider: bool = True
    hide_in_observations: bool = True


COLORS = {"cable": (0.80, 0.08, 0.06), "clip": (0.93, 0.93, 0.91), "clip_flange": (0.62, 0.62, 0.60),
          "table": (0.30, 0.31, 0.33), "gripper": (0.86, 0.86, 0.87), "gripper_finger": (0.16, 0.16, 0.18)}
# Calibrated so the table renders mid-grey (~150/255) and the cable saturated red.
DOME_INTENSITY = 150.0
KEY_LIGHT_INTENSITY = 600.0


class IsaacBackend:
    """Implements the SimBackend protocol of robot.executor plus scene/state/render methods."""

    def __init__(self, cable: CableSpec = CableSpec(), physics: PhysicsSpec = PhysicsSpec(),
                 headless: bool = True, render_resolutions: tuple = ((1600, 1200), (768, 768)),
                 rt_subframes: int = 4, robot: RobotSpec = RobotSpec(),
                 park_pose: tuple = (0.31, 0.0, 0.49, 0.0), viewport: tuple[int, int] = (1280, 720),
                 render_updates: bool = True):
        # Load this environment's torch (and its CUDA >= 12.8 runtime) before Kit starts.
        # Otherwise Isaac's deprecated ml_archive torch resolves libcudart.so.12 to the
        # node's system CUDA 12.4 via ldconfig, fails to import, and Kit shuts down.
        import torch  # noqa: F401
        from isaacsim import SimulationApp

        # Select the GPU with CUDA_VISIBLE_DEVICES; one GPU per simulator process.
        # CABLE_SIM_THREADS caps Kit's task and TBB threads (Isaac's default: min(cores, 16));
        # CABLE_PHYSX_THREADS sets the PhysX CPU dispatcher's worker count. Many simulator
        # processes per node run faster with small values than with the defaults.
        # Insertion training/evaluation never render: there the caller passes a small viewport and small
        # render products, so dozens of processes fit on one GPU. Physics is unaffected. (Kit needs the
        # render products to exist: a backend without cameras stalls.)
        launch = {"headless": headless, "renderer": "RealTimePathTracing", "multi_gpu": False,
                  "width": viewport[0], "height": viewport[1]}
        if os.environ.get("CABLE_SIM_THREADS"):
            launch["limit_cpu_threads"] = int(os.environ["CABLE_SIM_THREADS"])
        extra = []
        if os.environ.get("CABLE_PHYSX_THREADS"):
            extra.append(f"--/persistent/physics/numThreads={int(os.environ['CABLE_PHYSX_THREADS'])}")
        if os.environ.get("CABLE_RTX_DESCRIPTORS"):
            # RTX descriptors leak with every scene reload (load_scene); a process that reloads hundreds
            # of times (insertion training) needs a larger pool than Isaac's 900k reservation.
            n = int(os.environ["CABLE_RTX_DESCRIPTORS"])
            extra += [f"--/rtx/reservedDescriptors={n}", f"--/rtx/descriptorSets={max(4000, n // 3)}"]
        if extra:
            launch["extra_args"] = extra
        self.app = SimulationApp(launch)
        import omni.usd
        from isaacsim.core.rendering_manager import RenderingManager
        from isaacsim.core.simulation_manager import SimulationManager
        import isaacsim.core.experimental.utils.stage as stage_utils
        import isaacsim.core.experimental.utils.app as app_utils

        # Fabric mode: PhysX stops writing every link pose to USD each step (Kit's USD
        # listeners would otherwise dominate step time). Poses are copied into Fabric,
        # which the RTX renderer reads, only before a render (_sync_render).
        SimulationManager.enable_fabric(True)
        import omni.physxfabric

        self._fabric = omni.physxfabric.get_physx_fabric_interface()
        self._omni_usd = omni.usd
        self._SM = SimulationManager
        self._RM = RenderingManager
        self._app_utils = app_utils
        self.cable_spec = cable
        self.physics_spec = physics
        self.physics_dt = physics.dt
        self._render_resolutions = render_resolutions
        self._rt_subframes = rt_subframes
        self.robot_spec = robot
        self.park_pose = np.asarray(park_pose, dtype=float)
        stage_utils.create_new_stage()
        self.stage = omni.usd.get_context().get_stage()
        self._clip_ids: list[str] = []
        self._joint_path = "/World/GraspJoint"
        self._attached: int | None = None
        self._opening = 0.08
        self._playing = False
        self.robot = None
        self._build_static()
        self._build_cable()
        self._build_gripper()
        if robot.model != "none":
            from .robot_visual import FrankaVisual

            self.robot = FrankaVisual(self, robot)
        self._build_cameras()
        if not render_updates:
            self._disable_render_updates()

    # ------------------------------------------------------------------ authoring
    def _material(self, name: str, rgb) -> "object":
        from pxr import Gf, Sdf, UsdShade

        path = f"/World/Looks/{name}"
        mat = UsdShade.Material.Define(self.stage, path)
        shader = UsdShade.Shader.Define(self.stage, path + "/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return mat

    def _bind(self, prim, mat) -> None:
        from pxr import UsdShade

        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)

    def _build_static(self) -> None:
        from pxr import Gf, PhysxSchema, UsdGeom, UsdLux, UsdPhysics, UsdShade

        st = self.stage
        UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(st, 1.0)
        UsdGeom.Xform.Define(st, "/World")
        p = self.physics_spec
        scene = UsdPhysics.Scene.Define(st, "/World/PhysicsScene")
        scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
        scene.CreateGravityMagnitudeAttr(p.gravity)
        px = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
        px.CreateSolverTypeAttr(p.solver)
        px.CreateEnableGPUDynamicsAttr(False)
        px.CreateBroadphaseTypeAttr("MBP")
        px.CreateTimeStepsPerSecondAttr(int(round(1.0 / p.dt)))
        px.CreateEnableEnhancedDeterminismAttr(p.enhanced_determinism)
        self._SM.set_physics_dt(p.dt)

        self.mats = {k: self._material(k, v) for k, v in COLORS.items()}
        pm = UsdShade.Material.Define(st, "/World/PhysicsMaterial")
        api = UsdPhysics.MaterialAPI.Apply(pm.GetPrim())
        api.CreateStaticFrictionAttr(self.cable_spec.friction)
        api.CreateDynamicFrictionAttr(self.cable_spec.friction)
        api.CreateRestitutionAttr(0.0)
        self.phys_mat = pm

        table = UsdGeom.Cube.Define(st, "/World/Table")
        table.CreateSizeAttr(1.0)
        xf = UsdGeom.XformCommonAPI(table)
        xf.SetTranslate(Gf.Vec3d(0.0, 0.0, -0.01))
        xf.SetScale(Gf.Vec3f(8.0, 8.0, 0.02))
        UsdPhysics.CollisionAPI.Apply(table.GetPrim())
        self._bind(table.GetPrim(), self.mats["table"])
        UsdShade.MaterialBindingAPI.Apply(table.GetPrim()).Bind(
            pm, UsdShade.Tokens.weakerThanDescendants, "physics")

        dome = UsdLux.DomeLight.Define(st, "/World/DomeLight")
        dome.CreateIntensityAttr(DOME_INTENSITY)
        sun = UsdLux.DistantLight.Define(st, "/World/KeyLight")
        sun.CreateIntensityAttr(KEY_LIGHT_INTENSITY)
        sun.CreateAngleAttr(2.0)
        UsdGeom.XformCommonAPI(sun).SetRotate(Gf.Vec3f(15.0, -10.0, 0.0))

    def _physics_bind(self, prim) -> None:
        from pxr import UsdShade

        UsdShade.MaterialBindingAPI.Apply(prim).Bind(self.phys_mat, UsdShade.Tokens.weakerThanDescendants, "physics")

    def _build_cable(self) -> None:
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

        c, p = self.cable_spec, self.physics_spec
        st = self.stage
        UsdGeom.Scope.Define(st, "/World/Cable")
        UsdGeom.Scope.Define(st, "/World/CableJoints")
        n = c.n_links
        self.link_paths = [f"/World/Cable/link_{i:03d}" for i in range(n)]
        cyl = c.link_length - c.radius  # capsules overlap neighbours by one radius: no gaps
        mass = c.mass_per_m * c.link_length
        for i, path in enumerate(self.link_paths):
            cap = UsdGeom.Capsule.Define(st, path)
            cap.CreateHeightAttr(cyl)
            cap.CreateRadiusAttr(c.radius)
            cap.CreateAxisAttr("X")
            UsdGeom.XformCommonAPI(cap).SetTranslate(Gf.Vec3d(-i * c.link_length, 0.0, c.radius))
            prim = cap.GetPrim()
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.RigidBodyAPI.Apply(prim)
            UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
            col = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            col.CreateContactOffsetAttr(c.contact_offset)
            col.CreateRestOffsetAttr(c.rest_offset)
            rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            rb.CreateSolverPositionIterationCountAttr(p.position_iterations)
            rb.CreateSolverVelocityIterationCountAttr(p.velocity_iterations)
            rb.CreateLinearDampingAttr(c.linear_damping)
            rb.CreateAngularDampingAttr(c.angular_damping)
            rb.CreateMaxDepenetrationVelocityAttr(p.max_depenetration_velocity)
            self._bind(prim, self.mats["cable"])
            self._physics_bind(prim)
        for i in range(n - 1):
            j = UsdPhysics.Joint.Define(st, f"/World/CableJoints/j_{i:03d}")
            j.CreateBody0Rel().SetTargets([self.link_paths[i]])
            j.CreateBody1Rel().SetTargets([self.link_paths[i + 1]])
            j.CreateLocalPos0Attr(Gf.Vec3f(-c.link_length / 2, 0, 0))
            j.CreateLocalPos1Attr(Gf.Vec3f(c.link_length / 2, 0, 0))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0))
            jp = j.GetPrim()
            for dof in ("transX", "transY", "transZ", "rotX"):
                lim = UsdPhysics.LimitAPI.Apply(jp, dof)
                lim.CreateLowAttr(1.0)
                lim.CreateHighAttr(-1.0)
            for dof in ("rotY", "rotZ"):
                lim = UsdPhysics.LimitAPI.Apply(jp, dof)
                lim.CreateLowAttr(-c.bend_limit_deg)
                lim.CreateHighAttr(c.bend_limit_deg)
                drv = UsdPhysics.DriveAPI.Apply(jp, dof)
                drv.CreateTypeAttr("force")
                drv.CreateStiffnessAttr(c.bend_stiffness)
                drv.CreateDampingAttr(c.bend_damping)
                drv.CreateTargetPositionAttr(0.0)

    def _build_gripper(self) -> None:
        from pxr import Gf, UsdGeom, UsdPhysics

        from ..robot.gripper import GripperDims

        g = GripperDims()
        st = self.stage
        root = UsdGeom.Xform.Define(st, "/World/Gripper")
        UsdPhysics.RigidBodyAPI.Apply(root.GetPrim()).CreateKinematicEnabledAttr(True)
        UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(1.0)
        xf = UsdGeom.XformCommonAPI(root)
        xf.SetTranslate(Gf.Vec3d(*map(float, self.park_pose[:3])))
        bottom = -g.fingertip_below_grasp
        for name, center, size in (
            ("finger_pos", (0.0, 0.02, bottom + g.finger_length / 2), (g.finger_width, g.finger_thickness, g.finger_length)),
            ("finger_neg", (0.0, -0.02, bottom + g.finger_length / 2), (g.finger_width, g.finger_thickness, g.finger_length)),
            ("palm", (0.0, 0.0, bottom + g.finger_length + g.palm_size[2] / 2), g.palm_size),
        ):
            cube = UsdGeom.Cube.Define(st, f"/World/Gripper/{name}")
            cube.CreateSizeAttr(1.0)
            cx = UsdGeom.XformCommonAPI(cube)
            cx.SetTranslate(Gf.Vec3d(*center))
            cx.SetScale(Gf.Vec3f(*size))
            self._bind(cube.GetPrim(), self.mats["gripper_finger" if name.startswith("finger") else "gripper"])
        # Visual-only wrist and flange above the palm, so the floating hand reads as the end
        # of a robot arm in videos. No colliders: collision uses robot.gripper.gripper_boxes.
        palm_top = bottom + g.finger_length + g.palm_size[2]
        for name, radius, height in (("flange", 0.045, 0.015), ("wrist", 0.035, 0.30)):
            cyl = UsdGeom.Cylinder.Define(st, f"/World/Gripper/{name}")
            cyl.CreateRadiusAttr(radius)
            cyl.CreateHeightAttr(height)
            cyl.CreateAxisAttr("Z")
            z0 = palm_top if name == "flange" else palm_top + 0.015
            UsdGeom.XformCommonAPI(cyl).SetTranslate(Gf.Vec3d(0.0, 0.0, z0 + height / 2))
            self._bind(cyl.GetPrim(), self.mats["gripper"])
        self._gripper_dims = g

    def _build_cameras(self) -> None:
        import omni.replicator.core as rep
        from pxr import UsdGeom

        rep.orchestrator.set_capture_on_play(False)
        self._rep = rep
        self._cams = {}
        for name, res in zip(("full", "zoom"), self._render_resolutions):
            path = f"/World/Cameras/{name}"
            cam = UsdGeom.Camera.Define(self.stage, path)
            cam.CreateClippingRangeAttr((0.01, 50.0))
            rp = rep.create.render_product(path, tuple(res))
            ann = rep.AnnotatorRegistry.get_annotator("rgb")
            ann.attach([rp])
            self._cams[name] = (cam, rp, ann, tuple(res))

    def _disable_render_updates(self) -> None:
        """For processes that never render (insertion training/evaluation): stop the viewport and the
        render products from rendering on every app update (load_scene's updates included). Physics
        does not depend on rendering; render() is then unavailable."""
        import omni.kit.viewport.utility as vu

        for _, rp, _, _ in self._cams.values():
            rp.hydra_texture.set_updates_enabled(False)
        vp = vu.get_active_viewport()
        if vp is not None:
            vp.updates_enabled = False
        self._render_disabled = True

    # -------------------------------------------------------------- scene / state
    def load_scene(self, clips: list[Clip]) -> None:
        """(Re)create the static clip colliders. Stops and restarts the simulation."""
        from pxr import Gf, UsdGeom, UsdPhysics

        self.stop()
        for cid in self._clip_ids:
            self.stage.RemovePrim(f"/World/Clips/{cid}")
        UsdGeom.Scope.Define(self.stage, "/World/Clips")
        self._clip_ids = []
        for clip in clips:
            root = UsdGeom.Xform.Define(self.stage, f"/World/Clips/{clip.clip_id}")
            xf = UsdGeom.XformCommonAPI(root)
            xf.SetTranslate(Gf.Vec3d(clip.pose.x, clip.pose.y, 0.0))
            xf.SetRotate(Gf.Vec3f(0.0, 0.0, float(np.degrees(clip.pose.yaw))))
            for b in clip.boxes():
                cube = UsdGeom.Cube.Define(self.stage, f"/World/Clips/{clip.clip_id}/{b.name}")
                cube.CreateSizeAttr(1.0)
                cx = UsdGeom.XformCommonAPI(cube)
                cx.SetTranslate(Gf.Vec3d(*b.center))
                cx.SetScale(Gf.Vec3f(*b.size))
                UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
                self._bind(cube.GetPrim(), self.mats["clip_flange" if b.name.startswith("flange") else "clip"])
                self._physics_bind(cube.GetPrim())
            self._clip_ids.append(clip.clip_id)
        self.start()

    def start(self) -> None:
        import omni.physics.tensors as tensors

        if self._playing:
            return
        self._app_utils.play()
        self._RM.render()
        self._SM.initialize_physics()
        stage_id = self._omni_usd.get_context().get_stage_id()
        self._view = tensors.create_simulation_view("numpy", stage_id=stage_id)
        self._view.set_subspace_roots("/")
        self._links = self._view.create_rigid_body_view("/World/Cable/link_*")
        self._gripper = self._view.create_rigid_body_view("/World/Gripper")
        assert self._links.count == self.cable_spec.n_links, self._links.count
        self._all = np.arange(self._links.count, dtype=np.int32)
        self._g_idx = np.zeros(1, dtype=np.int32)
        self._playing = True

    def stop(self) -> None:
        if self._playing:
            self.detach()
            self._app_utils.stop()
            self._RM.render()
            self._playing = False

    def capture_state(self) -> dict:
        return {"link_transforms": self._links.get_transforms().copy(),
                "link_velocities": self._links.get_velocities().copy(),
                "cable_spec": asdict(self.cable_spec), "physics_spec": asdict(self.physics_spec)}

    def restore_state(self, state: dict) -> None:
        self.detach()
        self._links.set_transforms(np.asarray(state["link_transforms"], dtype=np.float32), self._all)
        self._links.set_velocities(np.asarray(state["link_velocities"], dtype=np.float32), self._all)
        self.set_gripper_pose(self.park_pose, 0.08)

    def set_polyline(self, points_head_first: np.ndarray) -> None:
        """Place the cable links along a head-first polyline sampled at the link length (at rest)."""
        pts = np.asarray(points_head_first, dtype=float)
        assert len(pts) == self.cable_spec.n_links + 1
        centers = 0.5 * (pts[:-1] + pts[1:])
        fwd = pts[:-1] - pts[1:]
        yaw = np.arctan2(fwd[:, 1], fwd[:, 0])
        tf = np.zeros((len(centers), 7), dtype=np.float32)
        tf[:, :3] = centers
        q = np.array([yaw_to_quat_wxyz(a) for a in yaw])  # wxyz
        tf[:, 3:6], tf[:, 6] = q[:, 1:], q[:, 0]  # tensor API uses xyzw
        self.detach()
        self._links.set_transforms(tf, self._all)
        self._links.set_velocities(np.zeros((len(centers), 6), dtype=np.float32), self._all)

    # ------------------------------------------------------- SimBackend protocol
    def cable_points(self) -> np.ndarray:
        return link_points(self._links.get_transforms(), self.cable_spec.link_length)  # tensor API: xyzw

    def point_segment(self, point_index: int) -> int:
        n = self.cable_spec.n_links
        return int(np.clip(point_index - 1 if point_index > 0 else 0, 0, n - 1))

    def set_gripper_pose(self, pose: np.ndarray, opening: float) -> None:
        tf = self._gripper_tf(pose)
        self._gripper.set_kinematic_targets(tf, self._g_idx)
        self._gripper.set_transforms(tf, self._g_idx)
        self._set_opening(opening)

    def _gripper_tf(self, pose: np.ndarray) -> np.ndarray:
        q = yaw_to_quat_wxyz(float(pose[3]))
        return np.array([[pose[0], pose[1], pose[2], q[1], q[2], q[3], q[0]]], dtype=np.float32)

    def _set_opening(self, opening: float) -> None:
        from pxr import Gf, UsdGeom

        self._opening = float(opening)
        off = opening / 2 + self._gripper_dims.finger_thickness / 2
        for name, s in (("finger_pos", 1), ("finger_neg", -1)):
            prim = self.stage.GetPrimAtPath(f"/World/Gripper/{name}")
            t = UsdGeom.XformCommonAPI(prim).GetXformVectors(0)[0]
            UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(t[0], s * off, t[2]))

    def gripper_pose(self) -> np.ndarray:
        """Current (x, y, z, yaw) of the kinematic grasp body."""
        x, y, z, qx, qy, qz, qw = self._gripper.get_transforms()[0].astype(float)
        return np.array([x, y, z, np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))])

    def _sync_render(self) -> None:
        """Copy current PhysX poses into Fabric and pose the arm (does not step physics)."""
        if self.robot is not None and self._playing:
            self.robot.update(self.gripper_pose(), self._opening, holding=self._attached is not None)
        self._fabric.update(self._SM.get_simulation_time(), self.physics_dt)

    def _write_usd_pose(self, path: str, tf: np.ndarray) -> None:
        """Author a body's current physics pose in USD (identical pose: PhysX state unchanged)."""
        from pxr import Gf, UsdGeom

        xf = UsdGeom.Xformable(self.stage.GetPrimAtPath(path))
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, tf[:3])))
        xf.AddOrientOp().Set(Gf.Quatf(float(tf[6]), float(tf[3]), float(tf[4]), float(tf[5])))

    def attach(self, segment: int) -> None:
        from pxr import Gf, UsdPhysics

        self.detach()
        g = self._gripper.get_transforms()[0].astype(float)
        l = self._links.get_transforms()[segment].astype(float)
        # Keep the kinematic gripper's USD pose current for the joint parser. Do not
        # write the link's pose: PhysX would apply it as a (same-pose) teleport, which
        # perturbs solver state and breaks bitwise-repeatable resets. PhysX may then log
        # "CreateJoint - found a joint with disjointed body transforms": it compares the
        # frames against the link's stale USD pose (Fabric mode). The frames below come
        # from live PhysX poses, so nothing snaps (measured: grasped link moves 0.0 mm).
        self._write_usd_pose("/World/Gripper", g)
        Rg, Rl = _quat_xyzw_to_mat(g[3:7]), _quat_xyzw_to_mat(l[3:7])
        rel_pos = Rg.T @ (l[:3] - g[:3])  # link origin in the gripper frame
        rel_rot = Rg.T @ Rl
        j = UsdPhysics.FixedJoint.Define(self.stage, self._joint_path)
        j.CreateBody0Rel().SetTargets(["/World/Gripper"])
        j.CreateBody1Rel().SetTargets([self.link_paths[segment]])
        j.CreateLocalPos0Attr(Gf.Vec3f(*rel_pos))
        j.CreateLocalRot0Attr(_mat_to_gf_quatf(rel_rot))
        j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        j.CreateLocalRot1Attr(Gf.Quatf(1.0))
        self._attached = segment

    def detach(self) -> None:
        if self._attached is not None or self.stage.GetPrimAtPath(self._joint_path):
            self.stage.RemovePrim(self._joint_path)
        self._attached = None

    def step(self, n: int = 1, gripper_target: np.ndarray | None = None) -> None:
        if gripper_target is not None:
            self._gripper.set_kinematic_targets(self._gripper_tf(gripper_target), self._g_idx)
        self._SM.step(steps=n)

    # ------------------------------------------------------------------ rendering
    def render(self, cam: TopDownCamera) -> np.ndarray:
        """Render ``cam`` without advancing physics (PhysX writes body poses back to USD)."""
        from pxr import Gf, UsdGeom

        if getattr(self, "_render_disabled", False):
            raise RuntimeError("this backend was built without render updates (insertion tasks)")
        name = "full" if tuple(cam.resolution) == self._cams["full"][3] else "zoom"
        usd_cam, _, ann, _ = self._cams[name]
        usd_cam.GetFocalLengthAttr().Set(float(cam.focal_mm))
        usd_cam.GetHorizontalApertureAttr().Set(20.955)
        usd_cam.GetVerticalApertureAttr().Set(float(cam.vertical_aperture_mm))
        xf = UsdGeom.XformCommonAPI(usd_cam)
        xf.SetTranslate(Gf.Vec3d(cam.center_xy[0], cam.center_xy[1], cam.height))
        xf.SetRotate(Gf.Vec3f(0.0, 0.0, -90.0))
        hide = self.robot_spec.hide_in_observations
        if hide:
            self.set_robot_visible(False)
        self._sync_render()
        self._rep.orchestrator.step(rt_subframes=self._rt_subframes, pause_timeline=True, delta_time=0.0)
        img = np.asarray(ann.get_data())[..., :3].copy()
        if hide:
            self.set_robot_visible(True)
        return img

    def set_robot_visible(self, visible: bool) -> None:
        """Show or hide the arm (or, without a robot, the floating gripper); physics unaffected."""
        from pxr import UsdGeom

        path = self.robot.root if self.robot is not None else "/World/Gripper"
        img = UsdGeom.Imageable(self.stage.GetPrimAtPath(path))
        img.MakeVisible() if visible else img.MakeInvisible()

    def close(self) -> None:
        self.app.close()


def _quat_xyzw_to_mat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _mat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Rotation matrix (column-vector convention) -> unit quaternion (w, x, y, z)."""
    t = np.trace(R)
    if t > 0:
        s = 2.0 * np.sqrt(t + 1.0)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
        q = [0.0, 0.0, 0.0, 0.0]
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.array(q)
    return q / np.linalg.norm(q)


def _mat_to_gf_quatf(R: np.ndarray):
    from pxr import Gf

    w, x, y, z = _mat_to_quat_wxyz(R)
    return Gf.Quatf(float(w), float(x), float(y), float(z))
