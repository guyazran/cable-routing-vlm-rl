# Cable-routing horizon check

This repository is the environment for the study specified in [`horizon-check.md`](horizon-check.md). The study asks whether a hierarchical VLM + RL cable-routing system becomes less reliable when the required route grows from **five to ten clips**. The system reconstructs arXiv:2510.19268.

**No experiment has been run yet.** Every run under `runs/smoke/` is an infrastructure or development check on development scenes.

Every reconstruction choice is listed in [`reports/deviation_log.md`](reports/deviation_log.md). All parameters live in [`configs/default.yaml`](configs/default.yaml), and its hash identifies a frozen study.

## Software layout

| Component | Where | Notes |
|---|---|---|
| Isaac Sim 6.2.0 (Kit 110.4, PhysX 110.4.4) | `~/opt/isaacsim/6.2.0`, symlinked as `~/opt/isaacsim/current` | User-global standalone install. `scripts/install/install_isaacsim.sh` downloads it, checks its MD5 and extracts it in parallel. |
| Franka Panda asset (Isaac Sim 6.2) | `assets/robots/franka_panda/` (git-ignored, 36 MB) | `python scripts/install/fetch_robot_assets.py` caches the USD and every layer/texture it uses, so rendering needs no network. |
| vLLM 0.32.0 (VLM server) | `~/.local/share/uv/tools/vllm`, with `vllm` on `PATH` | User-global uv tool with its own Python 3.12 and torch. It is kept separate from the simulator environment, as the protocol requires. |
| Qwen3.8-27B @ `1d4bf0f2…`, Qwen3.5-9B @ `c2022362…` | HF cache (`~/.cache/huggingface`) | Pinned revisions in `configs/versions.yaml`. |
| `cable` conda env | `~/miniforge3/envs/cable` | Python 3.12 (matching Kit), numpy 2.3.1 / scipy 1.17.0 / torch 2.11.0+cu128 (matching Isaac's bundle), SB3 2.9.2, gymnasium 1.4.0, openai client. This project is installed editable. |

`conda activate cable` exposes Isaac Sim through the activation hook written by `scripts/install/create_cable_env.sh`.

- **Kit paths.** The hook sources Isaac's `setup_python_env.sh` but drops Kit's own stdlib directory.
- **CUDA runtime.** It puts the env's CUDA 12.8 runtime ahead of the node's system CUDA 12.4. Otherwise Isaac's bundled torch fails to import and Kit shuts down.
- **EULA.** It sets `OMNI_KIT_ACCEPT_EULA=YES`, which means the Isaac Sim license is accepted.
- **pip.** Use `cable-pip install …` to add packages. It is pip with Isaac's paths hidden.

Why Isaac Sim 6.2 and not GarmentLab's 4.0.0: on this node's driver (595.58.03), Isaac Sim 4.x/5.x crash in the RTX renderer. 6.2 is tested on exactly this driver. GarmentLab also has no cable model. See deviation S1.

## Code layout

The algorithm under test and the simulated environment are separate packages. `src/dloroute` is the algorithm: the paper's hierarchical method plus the protocol's reference controller. It never imports the simulator or `cablerouting`. The environment core (`geometry`, `sim`, `scenes`, `robot`) never imports `dloroute`. Only the experiment harness (`bridge`, `eval`, `cli`, `config`) wires the two together. `tests/test_separation.py` enforces all three rules.

```text
configs/        default.yaml (every parameter), versions.yaml (pinned software/models)
prompts/        system.md, task.md, examples.json (+ examples/*.png, to render from dev scenes)
manifests/      scene manifests per family/pool (rejected seeds logged); states/ = settled initial states
reports/        deviation_log.md
scripts/        install/, serve_vlm.sh, smoke/ (isaac_hello, isaac_smoke, vlm_smoke, profile_step)

src/dloroute/          ALGORITHM (numpy; optional PIL/openai/gymnasium/SB3; no simulator)
  interface.py  the only contract with the world: SceneSpec/ClipSpec, CableState (centreline + transits),
                PickPlace command, Views, World protocol
  state.py      head metrics relative to a clip, route status (valid prefix, completion)
  decision.py   structured decision, JSON schema, syntactic validation (no silent correction)
  skills.py     Insert (frozen policy, closed loop, local horizon 7), Pull, Flatten -> pick-and-place commands
  rl/           observation, Eq. 1-4 reward, policies (SAC loader, heuristic), training MDP + curriculum, SAC
  planner/      prompt builder (memory policy), vLLM client (one format retry, failure classes)
  controllers/  VLM controller (+ true-progress diagnostic), ground-truth reference state machine
  agent.py      controller + skill library: one decision, one skill execution

src/cablerouting/      ENVIRONMENT (no algorithm imports)
  geometry/     clip model, threading evaluator (grader), gripper/clip collision
  robot/        pick-and-place executor, gripper model, Franka Panda kinematics (DH FK, top-down DLS IK)
  scenes/       N-clip route family generator (feasibility checks, shape motifs); shape x length
                train/test dataset (dataset.py); single-clip insertion start states
  sim/          Isaac Sim backend (capsule-chain cable, kinematic gripper, cameras), routing env, overlays
                      HARNESS (imports both)
  bridge.py     SimWorld / SimInsertionWorld: the environment as a dloroute World
  eval/         episode runner (B(N) = bN), progress tracking, screen analysis
  analysis/     Wilson / exact McNemar / discordance CI / paired bootstrap / S(k), h(k) / power
  cli.py, config.py
tests/          unit tests for everything simulator-independent (`python -m pytest`)
```

### Where the information boundaries sit

- **World → algorithm.** The algorithm receives clip poses and scales, the cable centreline, and the clip transits. In simulation the transits come from the environment's evaluator. It also receives rendered views and the answer to a grasp-collision query.
- **Algorithm → world.** The only action is a pick-and-place command: grasp a centreline index, carry it through planar waypoints, then release.
- **Who reads the cable state.** The VLM controller reads only views and its own action history. The cable state is read by the Insert skill (the paper's state-based policy), the reference controller, and the labelled true-progress diagnostic. `tests/test_algorithm.py` checks this with a world that refuses state reads.
- **Grading.** The episode runner grades with the environment's evaluator, independent of the algorithm's own route status. `tests/test_bridge.py` checks that the two agree on geometric cases.
- **A real robot.** It would implement the same `World` with perception and a motion stack, without changing `dloroute`.

## Robot

The simulated robot is a **Franka Emika Panda** at the virtual base (origin, facing +x), as in the paper's simulation media. See deviation P6.

- **Kinematic arm on the existing grasp body.** The cable physics is unchanged. Before every render, IK (`robot/franka.py`) puts the point 4 mm above the Panda fingertips on the kinematic grasp body, with a top-down hand whose closing axis is the gripper yaw. The arm itself has no bodies, joints or colliders.
- **Pedestal collider.** A static box at the pedestal footprint keeps the cable out of the base.
- **Hidden from observations.** Planner and crop renders hide the arm, as a wrist camera at the observation pose would not see it. The VLM's images are therefore unchanged.
- **Not in insertion tasks.** `train-insertion` and `eval-insertion` run without the robot, because their target clip sits at the origin, where the pedestal collider would be. `SimInsertionWorld` refuses a backend that has the robot. See deviation P6.
- **Reach.** Top-down at table height the Panda reaches 0.20–0.78 m. The study families' reach model (0.15–0.90 m) is wider, so some primary10 manipulation targets are outside a real Panda's reach. The video logs report the IK residual.
- **Layout families.** `hairpin6`, `scatter5` and `zigzag5` (deviation L5) have sharper turns and longer moves between clips. They use Franka-exact reach and keep 0.20 m clear of the base. They are development/demo families only.
- **Showcase.** `reports/robot_demo/showcase.html` is self-contained, built by `scripts/demo/build_showcase.py`. It holds videos, close-ups, layout schematics and outcomes. Full-resolution videos are in `runs/demo/robot/<scene>/routing.mp4`.
- **First outcomes (2026-11-02).** Physics-only runs with the reference controller and the heuristic stand-in Insert (`runs/demo/dry`):
  - New families: 1 of 24 development scenes completed.
  - primary10: 0 of 6.
  - Ways runs ended: Insert collision 19, Pull/Flatten collision 7, decision cap 4, complete 2.
  - Reasons: the heuristic Insert's gripper overlaps a clip, and at sharp turns Flatten loops while the bent cable springs back.

## Official dataset (shape × length)

The `shapes` dataset is the official data of the study. `horizon-check.md`, Section 8 ("Official data"), assigns each file its protocol role. It was generated on 2026-11-02 by `python -m cablerouting make-dataset` from the `dataset` section of `configs/default.yaml`. See deviation L6.

| File | Role | Scenes |
|---|---|---|
| `manifests/shapes_train.json` | development (tuning, training, examples) | 960, 20 per cell |
| `manifests/shapes_validation.json` | calibration pool (separate pool, own seed range) | 384, 8 per cell |
| `manifests/shapes_test.json` | evaluation, held out | 960, 20 per cell |
| `manifests/shapes_screen.json` | the screen's initial states (subset of test) | 30 ten-clip scenes, 5 per shape |
| `manifests/shapes_calibration_r1.json`, `_r2.json` | calibration rounds (subsets of validation) | 24 five-clip + 24 ten-clip each |
| `manifests/shapes_split.json` | split record: seeds, selection rules, scene IDs, manifest hashes, balance counts | — |

- **Grid.** 6 shapes × 8 lengths (3–10 clips) = 48 cells.
- **Train/test split.** Train and test come from one pool of 40 scenes per cell, split 20/20 by a seeded permutation (`split_seed`). Within each cell, each of the four mirror × reverse variants contributes exactly 5 train and 5 test scenes.
- **Validation.** Validation is generated as a separate pool: 8 scenes per cell, 2 per variant. Adding it leaves train and test unchanged (tested).
- **Subsets.** The screen and calibration subsets are fixed by rule from scene IDs and variants, never from outcomes; the rules are in the protocol.
- **Held out.** Do not render or review test layouts before the evaluation runs.
- **Shapes** (generator `motif`; every stage adds ±2° jitter, chords 16–19 cm as in primary10):

| Shape | Turn pattern per stage |
|---|---|
| `arc` | constant 15–22° |
| `circle` | constant 28–34°; closes into a loop on long routes |
| `spiral` | same sign, tightening linearly from 6–12° to 45–55° (loosening in reversed variants) |
| `serpentine` | ±22–32°, sign flips every 2–3 stages, plus a 6–12° backbone bias |
| `zigzag` | straight legs and 60–80° corners of alternating sign, plus a 4–10° backbone bias |
| `square` | 86–94° corners of one sign every 3–4 stages, straight in between |

- **Short routes keep their shape.** A 3-clip square still has a corner; a 4-clip zigzag has two alternating corners; every serpentine has a sign flip.
- **Construction.** Each shape is drawn whole, then rigidly placed (uniform rotation, then a translation that keeps it in view, accepted when the reach checks pass). Clip, cable, view and reach settings are primary10's, plus 0.20 m clearance from the robot base.
- **Seeds.** Every cell has its own seed range: from 10,000,000 for train/test and from 20,000,000 for validation. These are disjoint from each other and from the generator's development, calibration and evaluation pools.
- **Selection effect.** Rejection keeps the realised shape parameters close to the drawn ranges, except for long circles: at 9–10 clips the accepted per-stage turn averages 29.0–29.7° instead of the drawn 31°, so their loops are slightly more open. Per-cell rejection counts and reasons are in the split record.
- **Settled initial states** (2026-11-02). States exist for the screen, both calibration rounds and the train split: 1,086 in `manifests/states/`, 5 MB.
  - **Checks.** Every state passed the predeclared checks in `scenes/state_checks.py`: no clip threaded, head inside C01's insertion region, cable flat on the table and outside every clip, and within 2 cm of the generated polyline. The worst settling displacement was 1.2 cm; the median was 0.
  - **Index.** Each manifest's `manifests/states/<manifest>.index.json` holds the checks and a physics hash. `freeze` refuses states made under other physics settings, so retuning the physics means rerunning `make-states --force`.
  - **Not generated.** The other test scenes (for a later confirmation) and the validation scenes at other lengths have no states yet.
- **Insertion training uses the train split.** `train-insertion` draws route-stage starts from the train split's 5,280 stages, with turns of 0–95° (deviation R8).
- **Not yet done.** `run-episode --targets` accepts only 5 and 10.
- **Schematics.** `python scripts/demo/plot_dataset.py --split train|validation` (no simulator) draws one scene per cell. The test split is deliberately not drawable.

## How the environment supports raising the horizon

The five- and ten-target conditions differ **only** in the requested route and the budget. Everything physical and visual is shared.

- **One scene family with ten physical clips** (`primary10`). A scene's five-target route is the first half of its ten-target route: `scene.route(5) == scene.route(10)[:5]`.
- **The same physical setup in both conditions:** the same saved initial state, cable, camera framing, clip geometry, policy and prompts. Unrequested clips are named as obstacles in the five-target prompt.
- **One cable length (2.4 m) for every condition.** It leaves at least 0.30 m behind C01 after all ten insertions.
- **Constant insertion-policy observation.** It is sampled by arc length from the head, so it does not depend on cable length or route position.
- **Counterbalanced route difficulty.** Local turn motifs are counterbalanced, so the last five stages are not harder by construction. Per-stage distance, turn, reach margin and slack are archived.
- **Budget scaled with horizon.** B(N) = b·N skill invocations, with one completion query after the last.
  - Equal-prefix-budget attainment (b·k) is computed for early-progress comparisons.
  - Internal RL attempts, physical operations, queries, tokens and wall time are counted separately.
- **Geometric progress after every action.** The valid ordered prefix is tracked, with first attainment, regressions and repairs, and error events. This yields S(k), h(k) and failure mechanisms up to k = 10.
- **Planner context sized for the longest ten-clip history.** The longest history (100 entries plus 4 images) is part of the serving check. Cutoffs, format failures and context overflow are separate terminal categories.
- **Pause during model calls.** Physics does not step while the model is queried, so model latency cannot change the cable's settling.

## Usage

All commands run inside `conda activate cable`. Pin one GPU per simulator process with `CUDA_VISIBLE_DEVICES`. The planner uses GPUs 0 and 1.

```bash
python -m pytest                                   # unit tests (no simulator needed)
CUDA_VISIBLE_DEVICES=3 python scripts/smoke/isaac_smoke.py --out runs/smoke/isaac
scripts/serve_vlm.sh main &                        # Qwen3.8-27B, TP=2 on GPUs 0,1
python scripts/smoke/vlm_smoke.py --scene runs/smoke/isaac/full.png --crop runs/smoke/isaac/crop_C01.png --structured

python -m cablerouting gen-scenes --family primary10 --pool development --count 20
python -m cablerouting make-dataset                # official shapes dataset: splits + screen/calibration subsets
python scripts/demo/plot_dataset.py --split train --out runs/demo/dataset/shapes_train.png
python -m cablerouting make-states --manifest manifests/primary10_development.json
scripts/make_states_parallel.sh manifests/shapes_train.json 3   # 3 Isaac shards with a stall watchdog, then check + index
python -m cablerouting make-states --manifest manifests/shapes_train.json   # check all states + write index (no Isaac if all exist)
python -m cablerouting render-scene --manifest manifests/primary10_development.json --scene-id primary10-dev-0000
CUDA_VISIBLE_DEVICES=2 python scripts/demo/render_layouts.py --scenes primary10_development:0,1,2,3 paper5_development:0
CUDA_VISIBLE_DEVICES=3 python scripts/demo/record_video.py --scenes hairpin6_development:0 --out runs/demo/robot  # wide + close-up mp4
python scripts/demo/record_video.py --scenes scatter5_development:0,1 --no-video --out runs/demo/dry  # outcomes only
CUDA_VISIBLE_DEVICES=3 python scripts/smoke/robot_smoke.py --out runs/smoke/robot  # arm loads, matches FK, follows the grasp
CUDA_VISIBLE_DEVICES=2 python scripts/bench/sim_speed.py --out runs/bench/full.json  # physics rate, render FPS, skill times

# Experiment drivers: implemented, not yet run
python -m cablerouting train-insertion --out checkpoints/sac_seed0 --steps 6200
python -m cablerouting eval-insertion --policy checkpoints/sac_seed0/sac_final.zip --n 100 --out reports/insertion_eval.json
python -m cablerouting run-episode --manifest … --scene-id … --controller vlm --targets 10 --policy … --out runs/…
python -m cablerouting run-screen --manifest manifests/primary10_evaluation.json --policy … --out runs/screen
python -m cablerouting analyze --runs runs/screen --out analysis/screen
```

`run-screen` refuses to start until `configs/FROZEN` exists, unless `--allow-unfrozen` is passed for development.

## Known log noise

- **Stalled `make-states` shards.** On 2026-11-02, 3 of 4 parallel Isaac processes stopped making progress after 57–249 states while still using CPU. The same scenes then settled normally in a fresh process. `scripts/make_states_parallel.sh` restarts a shard with no new state for 3 minutes. Settling is order-independent (bitwise identical across processes), so restarts don't change any state.

- **`CreateJoint - found a joint with disjointed body transforms`** on every grasp.
  - In Fabric mode, link poses stored in USD are stale.
  - The grasp joint's frames come from live PhysX poses, so nothing snaps: at attachment the grasped link moves 0.0 mm.
- **`omni.isaac.ml_archive` torch import errors** if Kit is started *outside* the `cable` env (e.g. `~/opt/isaacsim/current/python.sh`).
  - Cause: the node's ldconfig points `libcudart.so.12` at the system CUDA 12.4.
  - Kit shuts down on that error. Inside `cable` the env's torch is loaded first.
  - For bare `python.sh` use, set `LD_LIBRARY_PATH=~/opt/isaacsim/current/exts/isaacsim.pip.nv/pip_prebundle/nvidia/cuda_runtime/lib`.
- **First Kit start on a node takes 1–3 minutes**, then about 55 s.
  - Shader and extension caches live on NFS (`~/.cache/ov`).

## Status against the protocol (Section 15 checklist)

**Done:**
- The tools above are installed and pinned.
- The simulator smoke checks pass:
  - headless RTX rendering;
  - cable integrity: link stretch below 0.05%;
  - save/restore: the restored state matches to 1e-10 m, and replaying the same primitive after a restore is bitwise identical (PhysX enhanced determinism);
    - **exception (found 2026-11-02):** the *first* episode after `load_scene` ends about 0.1 mm away from later replays of the same episode, which are bitwise identical to each other. Rendering neither steps nor perturbs physics. See `scripts/smoke/render_physics_check.py` and `runs/smoke/render_steps/`. Chaotic cable dynamics can turn that 0.1 mm into a different episode outcome, so run order within a process matters until this is fixed;
  - the pick-and-place primitive;
  - the evaluator on a cable threaded through a clip.
- The training environment passes Gymnasium's `check_env`, and SAC plumbing works.
- **Planner serving checks pass** (protocol Section 4, items 1–5; `reports/serving_check_2026-11-01.json`):
  - The pinned 27B model loads in BF16 with TP=2, using about 42 GB per GPU.
  - Both images influence the answer.
  - A full four-image request takes about 5k prompt tokens and about 30 s.
  - The longest ten-clip history (100 entries) takes about 6k prompt + 1k output tokens, well inside 32k.
  - Reasoning and the answer are separated, and schema-constrained output works with reasoning on.
- **Ten-clip feasibility (v1 layout only).** On `primary10-dev-0000`, the reference controller with the heuristic stand-in policy threaded all ten clips in order. It used 10 invocations, with no resets; the evaluator confirms `C01+ … C10+`. The trace is in `runs/smoke/feasibility10_dev0000/`. This is not evaluation data.
  - That run (17:13, 2026-11-01) used the first, arch-shaped generator. dev-0000 was regenerated as a route-family-v2 layout at 21:36.
  - On the v2 layout, the same controller stops at 8/10: the heuristic Insert's gripper hits a clip at C09 (`runs/demo/dry/primary10-dev-0000`, 2026-11-02).
  - Ten-clip feasibility on the current family is therefore **not established**.
- All simulator-independent logic is unit-tested.
- The development manifests and settled states exist.
- The official shape × length dataset exists: train, validation and test splits plus screen and calibration subsets, as manifests without settled states (see above).

**Pending, in protocol order:**
1. Request the authors' artifacts. Not yet sent.
2. Develop and tune physics and skill parameters on development scenes. Establish ten-clip physical feasibility.
3. Train and freeze the insertion checkpoint, and evaluate it on 100 held-out single-clip scenes. `train-insertion` already samples route-stage turns from the train split (deviation R8).
4. Render the in-context example images into `prompts/examples/`. Build the perception development set.
5. Run the perception check on the development image set (Section 4). The serving checks are done.
6. Select the per-clip budget `b` on development scenes.
7. Run the calibration gates on `manifests/shapes_calibration_r1.json`. The official pools and their settled states exist (see "Official dataset").
8. Create `configs/FROZEN` with the config, prompt and checkpoint hashes. Only then run the screen.
