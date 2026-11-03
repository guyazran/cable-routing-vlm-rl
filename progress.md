# Horizon-check run: progress record

## Resumed 2026-11-03 21:30 (current)

The user asked to resume the run, with a directive: give the authors' method a fair shot, and do not fix it. The code must replicate the paper as cleanly as its text allows.

**Audit against the paper (arXiv 2510.19268v2), 21:30–21:45.** Deviation **R15** in `reports/deviation_log.md`:
- **R13 withdrawn.** No skill vetoes its own grasp any more:
  - Insert executes the policy's deterministic action;
  - Pull and Flatten run their predefined motions, and the Flatten body grasp is fixed at 6 cm;
  - a gripper–clip collision ends the episode, as in the paper (Sec. IV-A2, Fig. 6b).
- **Prompt:** the global budget line ("k of B invocations") was removed. The paper does not give the planner a global budget.
- **Unchanged:** the reference controller keeps its own rule-level grasp checks (R14); it is our ground-truth baseline, not the paper's method. The policy checkpoint, the training recipe (R11, R12) and prompt v2's skill descriptions are also unchanged.
- **Hashes:**
  - code `33ab53969a31246b`;
  - config `95f654b8b1b52a6b`;
  - prompt `9cce2ebbfae53832`.
- **Tests:** 169 pass.

**Running**
- **Planner S1:** clair1 GPUs 0,1, port 8000, `max-num-seqs` 48.
- **Workers on clair1:** 12, on GPUs 2,3, all using S1. Log `runs/study/logs/pod_clair1_B.out`.
- **Public job 1477883** (nlp-ada-2, 4 GPUs, 112 CPUs): planner S2 on 2 GPUs and 26 workers. Log `runs/study/logs/pod_horizA_1477883.out`.
- **Queue order for every worker:** `calib2`, `screen`, `diag`, `dev5`.
- **Development check (`dev5`):**
  - 24 VLM five-target episodes on train scenes, output to `runs/study/dev/vlm_r7`;
  - 48 reference episodes (24 five-target, 24 ten-target), output to `runs/study/dev/ref_r7`.

**Development check (`dev5`), 22:00.** No harness bugs found. Every failure is a genuine terminal collision.
- Arc and circle: both controllers succeed on almost all scenes.
- Zigzag: 0% for both controllers.
- Square: also 0% in every earlier development batch, for both controllers.
- At ~90° corners, Flatten leaves the head curled about 10 cm in front of the clip, and the policy cannot insert from that state.
- The 20/24 gate is therefore unlikely to pass. Options 1–3 were put to the user at 22:00. Recommended: run the screen anyway and document the failed gate. **The screen will not start without the user's go-ahead if the gate fails.**

**Calibration round 2.** System recorded 22:02:59 in `runs/study/calibration_r2/system.json` (code `33ab53969a31246b`). 96 episodes enqueued at 22:03 in `runs/study/queue/calib2`, ahead of the development tasks.

---

## Record of the first run (stopped 21:22)

The run of `horizon-check.md` was stopped by user request at 21:22 on 2026-11-03.

## Status at stop

**Nothing is running.**
- The Slurm pod jobs have ended.
- The planner server on clair1 and every worker, watcher and scheduled reminder are stopped.
- All four clair1 GPUs are idle.

**Where the protocol stands**

| Protocol step | State |
|---|---|
| Feasibility | Done. Ten-clip routes complete end to end on train scenes. |
| Insertion policy: train, select, held-out test | Done. Frozen candidate `checkpoints/sac_v6_s1_14808.zip`. |
| Development (Section 9) | Converging, but the last change is **untested on routes**. |
| Calibration round 1 (`shapes_calibration_r1`) | **Failed the gate** on the reference controller. Report kept in `reports/calibration_r1_failed.json`. |
| Freeze (`configs/FROZEN`) | Not done. |
| Screen and diagnostic (test split) | **Not started.** No test-split routing episode has run, and no test layout was rendered. |

**Held-out data used so far**
- **Insertion test starts** (seeds 9,000,000–9,000,099). Used once, for the selected checkpoint and the heuristic baseline (see "Insertion policy").
- **Calibration round 1** (validation split). Used and failed. Round 2 (`shapes_calibration_r2`) is still fresh.

## Results so far

### Insertion policy

**Training** (SAC, 2 seeds × 24 simulators each, ≤ 20k environment steps).

The first five training variants failed. The working recipe needed two reconstruction choices, both in `reports/deviation_log.md`:
- **R11:** success is an absorbing state, with terminal reward r/(1−γ), and only when it does not collide.
- **R12:** 3k warm-up transitions from the heuristic baseline plus N(0, 0.05) noise.

Failed runs are archived in `runs/study/train_failed/` and `runs/study/train_v1_terminal_success/`.

**Selection** (rule R10, `reports/insertion_selection.json`). Chosen: **seed 1 @ 14,808 steps**.

| Selected checkpoint, validation | Clean insertions |
|---|---|
| Single-clip | 100/100 |
| Route-stage | 94/100, 1 collision |

Seed 0 @ 19,728 was a close second (0.965 vs 0.970).

**Held-out single-clip test**, 100 starts, run once (`runs/study/insertion_heldout/`):

| Policy | Clean successes | Collisions | Mean steps | Mean signed distance |
|---|---|---|---|---|
| SAC (selected) | **100/100** | 0 | 2.00 | 8.3 cm |
| Heuristic baseline | 95/100 | 5 | 1.94 | 4.0 cm |
| *Paper, Table I* | *87% (RL) / 45% (heuristic)* | | *3.86* | *2.59 cm* |

Insertion is clearly easier in this simulator than in the paper. The heuristic also scores 97/100 single-clip and 88/100 route-stage on validation starts. This must be stated in any report.

### Development on train-split routes

| Batch | Five-target routes | Ten-target routes | Main failure |
|---|---|---|---|
| Reference, rules v3, seed-0 checkpoint | 19/24 | 13/18 | policy collisions |
| Reference, selected checkpoint | 12/15 | 7/12 | policy grasps beside the previous clip |
| Reference, + Insert grasp check that aborts (R13 as first written) | 14/24 | 9/24 | **loops to the budget** (9 + 14) |
| VLM, prompt v1 | 2/6 | 1/6 | Flatten or Pull next to clips |
| VLM, prompt v2 | 6/12 | 4/11 | collisions |

### Calibration round 1

Started 15:21. Interim report in `reports/calibration_r1_failed.json`; the system's hashes are in `runs/study/calibration_r1/system.json`.

| Check | Reference | VLM |
|---|---|---|
| 1: five-clip scenes, 5 requested | 14/24, with 9 budget exhaustions: **fail** | 10/12 run |
| 2: ten-clip scenes, first 5 requested | 13/23, with 10 budget exhaustions | 10/11 run |

**Why it failed:** the Insert grasp check, as first written, ends Insert without moving when the deterministic policy's grasp would hit a clip. In the same state the policy repeats the same action, so the reference controller loops until its budget runs out. The 26 episodes that had not started were withdrawn once check 1 could no longer pass.

## Code state

- **Not committed.** The repository still has no commits. Everything below is in the working tree.
- **Added this session**
  - `scripts/study/`: `pod.sh`, `train.sh` (with a watchdog), `enqueue.py`, `watch_checkpoints.sh`, `select_checkpoint.py`, `calibration_report.py`, `status.py`.
  - `scripts/dev/`: `render_examples.py`, `perception_check.py`.
  - `scripts/bench/thread_bench.py`.
  - `src/cablerouting/eval/workqueue.py` and the `worker`, `queue` and `freeze --calibration` commands.
  - Parallel and resumable SAC training, guided warm-up, absorbing success.
  - Reference rules v3, collision-aware primitives, prompt v2, the in-context example images.
- **Last change: implemented and unit-tested (168 tests pass), never run on a route.**
  - `SACPolicy.sample` in `dloroute/rl/policies.py`.
  - Insert resampling in `dloroute/skills.py` (`insert_resamples: 32` in the config). When the deterministic grasp would collide, Insert draws from the policy's own action distribution, seeded by the observation, until a grasp clears.
  - This is the intended fix for the calibration-1 loop. It changes the code and config hashes recorded for calibration round 1.
- **Configuration:**
  - Config hash `eb01ac61338f70d5` at calibration start; it changed afterwards with `insert_resamples`.
  - Prompt hash `b01e77d9d080761f`.
- **Documentation:**
  - `reports/deviation_log.md`: entries R9–R14, X1–X3, V6–V8, E1.
  - `reports/preregistration_2026-11-03.md`: the calibration and screen plan, the rerun rule, and the diagnostic, declared to run regardless of the screen's outcome.

## To resume

1. **Check the resampling fix on development routes.**
   - The queue `runs/study/queue/dev4ref` still holds 48 `p20_r6_*` reference tasks, writing to `runs/study/dev/ref_r6`.
   - Release stale claims first: `python -m cablerouting queue --queue runs/study/queue/dev4ref --reap`.
   - Then start a reference pod. Adding a few planner development episodes is advisable.
2. **If development results are credible, run calibration round 2** on fresh validation scenes:
   - `python scripts/study/enqueue.py schedule --queue runs/study/queue/calib2 --manifest manifests/shapes_calibration_r2.json --controllers vlm,reference --targets 5 --policy checkpoints/sac_v6_s1_14808.zip --out runs/study/calibration_r2`
   - Record `system.json` as for round 1, then check the gate (≥ 20/24 per check and controller) with `scripts/study/calibration_report.py`.
3. **If the gate passes:**
   - Freeze: `python -m cablerouting freeze --policy checkpoints/sac_v6_s1_14808.zip --manifests manifests/shapes_screen.json manifests/shapes_calibration_r2.json --calibration runs/study/calibration_r2/system.json`.
   - Then queue the screen (`schedule … --manifest manifests/shapes_screen.json --controllers vlm,reference --targets 5,10 --out runs/study/screen`) and the diagnostic (`--controllers vlm+progress --targets 10 --out runs/study/diagnostic`).
   - Analyse with `python -m cablerouting analyze`.
4. **If round 2 also fails:** generate more validation scenes under the validation seed scheme before the next attempt (protocol Section 9).

**Rough durations:**
- development check: about 20 min;
- calibration: about 1 h with 2 planner servers;
- screen plus diagnostic: about 1,900 planner decisions, about 3 h with 2 servers and about 22 planner workers.

## Operational notes

**Slurm limits**
- The public partition allows **300 CPUs and 4 GPUs per user**.
- clair1 (lab partition, cannot be preempted) has only the user's 4 GPUs and 32 cores.
- Public jobs on lab-owned nodes are preempted without warning. This happened on the nlp-pro6000 nodes.

**Isaac Sim at scale** (memory: `cluster-scaling-findings`)
- Use 4 Kit task threads and 1 PhysX thread per process: about 3.2 cores, with bitwise-identical physics.
- Fewer than 4 task threads deadlocks Kit.
- Processes that never render need 64×64 render products.
- Limits per GPU: about 12–14 processes, or about 7 rendering processes on a 24 GB L4.
- `omni.rtx: Out of resource descriptors` crashes still happen. `train.sh` resumes automatically.
- Spawned children need `LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6`.

**Episode outcomes and the planner**
- An episode's outcome can depend on its worker process's history: 1 of 3 re-runs differed (E1). The pairing protocol tolerates this.
- One planner server (TP = 2, 2× L40S) gives about 100 s per decision at 12 concurrent episodes, or about 430 decisions per hour.

**Stale queues.** These queues hold claims from killed pods: `dev2`, `dev3`, `calib` (round 1, withdrawn), `devvlm*` and `dev4vlm`. Start new queues, or run `--reap`, before reuse.
