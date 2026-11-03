# Pre-registration: horizon-check calibration and screen

Written 2026-11-03, before any calibration (validation-split) or screen (test-split) episode was run. Everything here follows `horizon-check.md`. Reconstruction choices are listed in `reports/deviation_log.md`.

## System under test

The configuration is frozen at calibration time. `configs/FROZEN` records it once the gate passes.

- **Insertion policy.**
  - Two SAC seeds were trained (recipe v6: deviations R9–R12).
  - The checkpoint is selected by `scripts/study/select_checkpoint.py` using the rule in R10: highest mean clean success over 100 single-clip and 100 route-stage validation starts, with ties broken by fewer collisions, then the earlier checkpoint.
  - The selection and its full candidate table are written to `reports/insertion_selection.json`.
  - The selected checkpoint is then evaluated **once** on the 100 held-out single-clip starts (seeds 9,000,000+).
- **Reference controller.**
  - `dloroute/controllers/reference.py` rules v2: grasp-collision checks in rules 3, 4 and 6.
  - Alignment tolerance stays 60°: on 48 development routes, 100° did no better (`runs/study/dev/ref_r2_align*`).
- **Budget.** b = 10 skill invocations per requested clip, so B(5) = 50 and B(10) = 100. No successful development episode came near the budget.
- **Planner.**
  - Qwen3.8-27B at the pinned revision, with the sampling profile and memory policy in `configs/default.yaml`.
  - Prompts and the in-context example images in `prompts/`.
  - The perception check passed on development crops (`runs/dev/perception/results.json`).

## Calibration (Section 9)

- **Scenes.** `manifests/shapes_calibration_r1.json`:
  - 24 five-clip validation scenes with all five clips requested (check 1);
  - 24 ten-clip validation scenes with the first five requested (check 2).
- **Controllers.** Both controllers run on the same initial states, with paired episode seeds (1000 × scene index).
- **Order.** Condition order within each scene is randomised with scheduling seed 20261031.
- **Gate.** At least 20/24 per check and controller (`scripts/study/calibration_report.py`).
- **If the gate fails:** go back to development on the train split, keep the failed report, and use round 2 (`shapes_calibration_r2.json`) for the next version.

## Screen (Section 10)

- **Scenes and conditions.** `manifests/shapes_screen.json`, 30 ten-clip test scenes. Four conditions: VLM ×5, VLM ×10, reference ×5, reference ×10. That is 120 episodes in the randomised schedule (`runs/study/screen/schedule.json`).
- **Progress-information diagnostic (Section 12), declared here before the screen.**
  - One additional condition, VLM + true progress ×10, on all 30 ten-target scenes, successes and failures alike.
  - It uses the same seeds as the screen's ten-target VLM episodes and is stored separately in `runs/study/diagnostic/`.
  - It runs **regardless of the screen's outcome**.
- **No outcome-based stopping or adaptation.** Prompts, checkpoint and budgets do not change after the screen starts.
- **Analysis (Section 12).**
  - Primary test: VLM five- vs ten-target full-task success, a two-sided exact McNemar test at α = 0.05.
  - Interval: Bonferroni Clopper–Pearson for the paired difference.
  - Effect of practical interest: 20 percentage points.
  - Everything else is diagnostic: the reference controller's decrease, I, S(k), h(k), prefix-budget attainment and failure mechanisms.

## Rerun rule (unscorable episodes)

An episode is unscorable if it crashes, loses its worker (preemption or node failure), or ends on a planner transport error. It is kept as `<episode>.attempt-<k>` with its cause, and rerun from its saved initial state, up to 4 attempts, independent of outcome (`src/cablerouting/eval/workqueue.py`). Model format errors, cutoffs and context overflows are outcomes, never reruns.

## Execution notes

- **Physics variation.** Episodes run in parallel on several hosts. Physics is bitwise reproducible for a fixed sequence of calls within one process. Across processes, the first-episode-after-load caveat (README) can change chaotic outcomes: one development scene out of three differed on re-execution. Each episode's worker, host and timestamps are logged.
- **Planner servers.** They are replicas of one pinned configuration (V7). Each episode records its server URL.
