"""Profile per-step costs of held gripper motion (development diagnostic)."""
import cProfile, pstats, time
import numpy as np
from cablerouting.config import load_config
from cablerouting.scenes.generator import generate_pool
cfg = load_config(); fam = cfg.family("primary10")
scene = generate_pool(fam, "development", 1)[0][0]
from cablerouting.sim.isaac_backend import IsaacBackend
sim = IsaacBackend(cfg.cable, cfg.physics, render_resolutions=(cfg.render.full_resolution, (768, 768)))
from cablerouting.sim.routing_env import RoutingEnv
from cablerouting.robot.executor import PickPlace, Via
env = RoutingEnv(sim, scene, fam.view_x, fam.view_y, cfg.motion, cfg.render)
sim.set_polyline(scene.initial_polyline(cfg.cable.link_length)); sim.step(240)
for name, fn in [("SM.step(1)", lambda: sim.step(1)), ("step(1,target)", lambda: sim.step(1, np.array([-0.6, 0, 0.3, 0.0])))]:
    t = time.perf_counter(); [fn() for _ in range(200)]; print(f"[prof] {name}: {(time.perf_counter()-t)/200*1e3:.2f} ms", flush=True)
import omni.physx
si = omni.physx.get_physx_simulation_interface()
t = time.perf_counter(); tt = 0.0
for _ in range(200): si.simulate(cfg.physics.dt, tt); si.fetch_results(); tt += cfg.physics.dt
print(f"[prof] raw simulate+fetch: {(time.perf_counter()-t)/200*1e3:.2f} ms", flush=True)
t = time.perf_counter(); [env.render("C01") for _ in range(3)]; print(f"[prof] render (both, x3): {(time.perf_counter()-t)/3:.2f} s each", flush=True)
c1 = env.clips[0]; p0 = sim.cable_points()
goal = c1.pose.to_world(np.array([c1.pose.to_local(p0[2])[0] + 0.06, 0.0]))
pr = cProfile.Profile(); pr.enable()
env.executor.pick_place(PickPlace(2, [Via(goal[0], goal[1], c1.pose.yaw)], 0.006, "prof"))
pr.disable(); pstats.Stats(pr).sort_stats("cumulative").print_stats(18)
sim.close()
