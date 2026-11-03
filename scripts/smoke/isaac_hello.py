"""Minimal Isaac Sim check: headless start, physics step, RTX camera render."""
import sys, time
t0 = time.time()
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
print(f"[hello] SimulationApp up in {time.time() - t0:.1f}s", flush=True)
import numpy as np
import omni.replicator.core as rep
import omni.usd
from pxr import Gf, UsdGeom, UsdLux, UsdPhysics
import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.rendering_manager import RenderingManager
from isaacsim.core.simulation_manager import SimulationManager

stage_utils.create_new_stage()
st = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
UsdPhysics.Scene.Define(st, "/World/PhysicsScene")
cube = UsdGeom.Cube.Define(st, "/World/Cube"); cube.CreateSizeAttr(0.1)
UsdGeom.XformCommonAPI(cube).SetTranslate(Gf.Vec3d(0, 0, 1.0))
UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim()); UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
cube.CreateDisplayColorAttr([(0.9, 0.1, 0.1)])
UsdLux.DomeLight.Define(st, "/World/Dome").CreateIntensityAttr(1000)
cam = UsdGeom.Camera.Define(st, "/World/Cam")
UsdGeom.XformCommonAPI(cam).SetTranslate(Gf.Vec3d(0, 0, 3.0))
rep.orchestrator.set_capture_on_play(False)
rp = rep.create.render_product("/World/Cam", (320, 240))
ann = rep.AnnotatorRegistry.get_annotator("rgb"); ann.attach([rp])
SimulationManager.set_physics_dt(1 / 240)
app_utils.play(); RenderingManager.render()
t1 = time.time(); SimulationManager.step(steps=240); print(f"[hello] 240 physics steps in {time.time()-t1:.2f}s", flush=True)
import omni.physics.tensors as tensors
view = tensors.create_simulation_view("numpy", stage_id=omni.usd.get_context().get_stage_id())
rb = view.create_rigid_body_view("/World/Cube")
print("[hello] cube z after 1 s free fall:", float(rb.get_transforms()[0, 2]), "(expected ~1.0-4.9 -> negative)", flush=True)
print("[hello] USD z:", UsdGeom.Xformable(cube).ComputeLocalToWorldTransform(0).ExtractTranslation()[2], flush=True)
z0 = float(rb.get_transforms()[0, 2]); t2 = time.time()
rep.orchestrator.step(rt_subframes=4, pause_timeline=True, delta_time=0.0)
print(f"[hello] first capture {time.time()-t2:.2f}s; physics moved during capture: {float(rb.get_transforms()[0, 2]) - z0:+.6f}", flush=True)
t2 = time.time(); rep.orchestrator.step(rt_subframes=4, pause_timeline=True, delta_time=0.0)
print(f"[hello] second capture {time.time()-t2:.2f}s; moved {float(rb.get_transforms()[0, 2]) - z0:+.6f}", flush=True)
img = np.asarray(ann.get_data())
print("[hello] image", img.shape, img.dtype, "mean", img[..., :3].mean(), flush=True)
from PIL import Image; Image.fromarray(img[..., :3]).save(sys.argv[1] if len(sys.argv) > 1 else "/tmp/hello.png")
print("[hello] OK", flush=True)
app.close()
