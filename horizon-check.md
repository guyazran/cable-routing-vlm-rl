# Cable-routing horizon check

Protocol date: 2026-10-31. Target paper: **Hierarchical DLO Routing with Reinforcement Learning and In-Context Vision-Language Models**, arXiv:2510.19268v2, revised 2026-04-15. [Paper][paper]

Methodology review: 2026-10-31. The 120-episode design below is an initial screen. A fixed larger sample is specified for a confirmatory study, and a targeted diagnostic is required before attributing a drop specifically to progress monitoring.

Data designation: 2026-11-02. The official scenes for this reproduction are the `shapes` dataset in `manifests/`, described in Section 8 under "Official data". On that date, before any evaluation episode, this reproduction chose to run the screen first (Section 12) and to calibrate on a separate validation split (Section 9).

This document specifies a simulation reproduction and a controlled test of routing through five versus ten clips. It is a plan, not an experiment report: the simulator, policies, prompts, and serving configuration still need to be implemented or validated. No results or measured runtimes are claimed here.

## 1. Objective and scope

Determine whether a hierarchical cable-routing system becomes less reliable when the required route grows from five to ten insertions, and identify the mechanisms responsible. Test the hypothesis without assuming that a degradation will occur.

The initial experiment uses a local open-weight VLM, one frozen RL insertion policy, scripted Pull and Flatten skills, and a ground-truth state reference controller. It requires no paid GPT-5 usage, real robot, SAM2 pipeline, or VLM fine-tuning.

There are two distinct questions:

1. **Practical reliability:** Does full-task success decrease at ten clips?
2. **Source of the decrease:** Is the change consistent with accumulating execution risk, worsening physical constraints, or additional VLM planning and monitoring errors?

Replacing the paper's GPT-5 planner means this is a reproduction of its architecture with an open-weight planner. Results apply directly to this implementation. They do not establish how the original GPT-5 implementation would perform at ten clips. A second model family would strengthen a later claim about the architecture more broadly.

The test can find a practical reliability limit; it cannot guarantee that ten clips will expose one. Explicit clip IDs and a supplied route simplify the paper's high-level planning problem. A positive result would expose a limitation even under that simplification. A negative result would not establish robustness to implicit goals, ambiguous attributes, or arbitrary routing geometries.

### Minimum viable study and decision points

1. **Feasibility:** Establish executable cable dynamics, all three skills, a working local VLM, and a geometrically valid ten-clip route on development scenes. If these are unavailable, the project is still a reconstruction effort, not a horizon experiment.
2. **Calibration:** Pass fresh five-target checks for both controllers on the validation split and check the components on difficult local states. Resolve implementation failures before freezing the system.
3. **Screen:** Run all 120 episodes on the 30 paired initial states of `manifests/shapes_screen.json`, with no outcome-based stopping. This can identify large effects and candidate mechanisms.
4. **Diagnosis:** If making a progress-monitoring claim, run the prespecified progress-information intervention in Section 12 on all 30 ten-target scenes, including the original successes.
5. **Confirmation, if needed:** Use a separately declared study on 114 fresh ten-clip test scenes (456 episodes; Sections 8 and 12) for a reasonably sensitive test of a 20-percentage-point drop. Do not keep adding scenes to the screen until a significant result appears.

## 2. What the paper establishes

The approximately 92% headline result concerns simulation. The reported real-robot success rate is 62.5%. The following values come from Tables I–III of the supplied v2 paper. [Paper][paper]

| Experiment | Reported success | Reported evaluation size |
| --- | ---: | --- |
| Single-clip insertion, heuristic | 45% | 100 scenes |
| Single-clip insertion, RL | 87% | 100 scenes |
| Long-horizon simulation, implicit order, full method | 80% | 15 trials |
| Long-horizon simulation, spatial order, full method | 93% | 15 trials |
| Long-horizon simulation, attribute order, full method | 93% | 15 trials |
| Long-horizon simulation, combined 4/5-clip setting, full method | 100% | 15 trials |
| Real robot, fixed-order controller with RL | 37.5% | 8 configurations |
| Real robot, full method | 62.5% | 8 configurations |

The combined 4/5-clip result is not a separate, extensive five-clip benchmark. The split between four- and five-clip trials needs clarification. Small evaluation samples and the absence of a ten-clip evaluation leave longer-horizon reliability unresolved.

The paper's symbolic planner also reaches 100% in the combined 4/5-clip setting. Its recovery ablation removes both recovery instructions and the Flatten skill, so that comparison does not isolate reasoning alone. Our primary comparison gives both controllers access to all three skills.

## 3. Obtain artifacts and record reconstruction choices

As checked on the protocol date, the authors' public repository contains their project website and media. No runnable training package, insertion checkpoint, or simulator configuration was identified there. GarmentLab is available separately, but it is not a ready-made implementation of this cable-routing task. [Authors' repository][authors-code] · [GarmentLab][garmentlab]

Before rebuilding, request or locate the following artifacts:

- Cable, clip, gripper, and scene assets, including dimensions, collision geometry, and attachment behavior.
- Isaac Sim/GarmentLab versions and the authors' modifications.
- Cable material parameters, particle representation, physics timestep, solver settings, and reset code.
- Insertion checkpoint, observation preprocessing, action decoding, SAC configuration, training seeds, and the meaning of a training “step.”
- Pull and Flatten implementations, including grasp selection and trajectory parameters.
- Complete planner prompts, example images, crop rules, action-history format, and model settings.
- Evaluation layouts, initial cable states, random seeds, success predicates, retry limits, and termination budgets.
- The four-/five-clip trial split and raw outcomes behind Table II.

Missing artifacts do not prevent a reconstruction. Record each substitute in a deviation log with its rationale and the experiments it affects. Do not describe reconstructed defaults as settings reported by the authors.

Time-box artifact discovery and a basic simulator/skill feasibility check before committing to the full rebuild. A mocked skill executor or image-only planner benchmark can test interface logic, but cannot support conclusions about closed-loop cable-routing reliability. Never substitute synthetic random skill failures for the physical task in the primary study.

## 4. Hardware and local VLM

Use **Qwen/Qwen3.8-27B** as the main experimental planner. It provides open weights under Apache 2.0, native image understanding, and documented local serving options. Its suitability for cable routing must be established on development scenes. [Model card][qwen]

| Component | Initial allocation | Notes |
| --- | --- | --- |
| Main VLM | Two L40S GPUs on the same host, BF16, tensor parallel size 2 | Approximately 54–56 GB for weights alone; runtime memory must be measured. |
| Simulator and insertion policy | A separate L40S | Reserve GPU memory for physics and rendering. |
| Optional development VLM | Qwen/Qwen3.5-9B, BF16, one L40S | Useful for debugging requests and orchestration; use the main model for the primary study. |
| Host resources | Start with 64 GB RAM and several hundred GB of SSD space | Planning allocation, not a measured requirement. |

L40S has 48 GB VRAM, supports BF16, and has no NVLink. Measure serving latency over its PCIe interconnect; do not infer throughput from weight fit alone. If only two GPUs are available, validate a resource schedule that also leaves the simulator sufficient memory. [NVIDIA specifications][l40s]

Pin the model commit, tokenizer/processor revision, serving-engine version or container digest, driver, CUDA, PyTorch, simulator, and asset versions. GarmentLab's public instructions target Isaac Sim 4.0.0; confirm compatibility instead of assuming the newest simulator is a drop-in replacement. Keep simulator and VLM environments separate.

### Candidate serving configuration

The following is a starting configuration to validate, not a command tested on our hardware. Install a compatible serving environment first and replace the model revision placeholder with a real commit.

```bash
export HORIZON_MODEL_REVISION="REPLACE_WITH_PINNED_MODEL_COMMIT"
CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen3.8-27B \
  --revision "$HORIZON_MODEL_REVISION" \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --reasoning-parser qwen3 \
  --mm-encoder-tp-mode data \
  --max-model-len 32768 \
  --max-num-seqs 2 \
  --limit-mm-per-prompt '{"image": 4}' \
  --gpu-memory-utilization 0.90 \
  --host 127.0.0.1 \
  --port 8000
```

The model documentation links serving recipes, but the current vLLM recipe explicitly says its verified coverage is text serving. Multimodal requests and this two-L40S configuration therefore need an explicit smoke check. SGLang is a fallback if the selected vLLM build cannot serve the images correctly. Do not enable a language-model-only mode. [Model documentation][qwen] · [vLLM recipe][vllm]

Before integrating the planner:

1. Verify that the actual model and revision were loaded in BF16 across the intended GPUs.
2. Submit a scene and crop containing different, easily checked information. Confirm that both images affect the response.
3. Submit the complete request with the two example images plus scene and crop. Increase the image-count allowance if the chosen prompt uses additional images.
4. Check that reasoning and final responses are separated and that structured decisions can be parsed.
5. Exercise the longest expected ten-clip history. Measure peak memory, prompt tokens, output tokens, latency, and completion finish reasons.

Also create a small development image set covering uninserted, partially inserted, fully inserted, occluded, and previously inserted-but-now-unthreaded states, with clips at the actual full-view and crop resolutions. Measure state recognition and premature-completion errors. A text-serving check or an answer about an unrelated image does not validate the perception needed by this task. Adjust image preprocessing or the model using development data if this check exposes a basic capability gap; preserve the results and freeze the final choice.

Start with medium reasoning effort. A candidate generation profile is temperature 1.0, top-p 0.95, top-k 20, an 8,192-token completion cap, and a fixed random seed where supported. Validate it during development, then freeze it. Qwen's reasoning levels are not calibrated to GPT-5's levels. Do not change reasoning effort between five- and ten-clip conditions.

The 32,768-token context and completion cap are initial allocations. Enlarge them before freezing the experiment if needed. Account for image tokens and reasoning tokens. A ten-clip result caused by silent truncation or missing final answers is a serving-budget issue that must be identified separately.

Pause simulated physics while waiting for a model decision, or use an identical fixed observation-to-action delay for both controllers. Otherwise model latency changes cable settling and introduces an uncontrolled physical difference from the faster reference. Log this timing policy.

## 5. Build and validate the manipulation environment

Implement the tabletop, clips, cable, gripper, camera, and pick-and-place execution in Isaac Sim/GarmentLab. Prefer the authors' assets and physical settings when available.

Record cable length, diameter, mass, stiffness, damping, friction, particle count and spacing, endpoint anchoring, self-collision behavior, and gripper attachment/release behavior. Record clip dimensions, opening orientation, collision meshes, and robot reach constraints. Grasping must not teleport or reset the cable into a favorable state between skills.

Provide these interfaces:

- Reset from a saved initial state, including the cable configuration and relevant simulator/controller state.
- Render a fixed full-scene view and a crop around a specified clip.
- Return the insertion policy's state observation in the target clip's local frame.
- Execute Insert, Pull, or Flatten and log the low-level trajectory.
- Evaluate insertion, routing order, collisions, and loss of previously established threading from simulator geometry.

Verify that saved states reset consistently and that the geometric evaluator distinguishes approaching a clip, partial insertion, full insertion, passing beside the clip, and later unthreading. A cable head beyond an exit plane alone is insufficient if it never passed through the opening. Validate the ordered cable centerline's passage through the clip aperture and its current threading, using directed entry/exit crossings and the cable's order from anchored end to free head. Record any dependence on clip topology or discretization. Check evaluator outputs against manually reviewed development traces; a coordinate threshold alone must not define success.

Check that robot/gripper collision failures and valid cable–clip contact are distinguished. Match the paper's collision termination convention where possible and document its exact implementation.

Evaluate progress after gripper release and a fixed, recorded number of settling simulation steps. Apply that delay identically after every skill for both controllers. A temporary insertion maintained only by the gripper must not count as retained threading.

### Train or load the insertion policy

Use one compatible insertion checkpoint for all primary conditions. If reconstructing the policy, use SAC with MLP actor and critic networks. The paper's experimental insertion action has seven parameters: one grasp index and two planar waypoints, each containing x, y, and rotation about z. One primitive grasps the indexed segment, follows both waypoints while holding it, then releases; do not introduce an extra regrasp between waypoints unless specified by the recovered implementation.

The observation includes ordered cable particles, clip pose and scale, and the partial-insertion indicator. Transform observations and decoded actions into the target clip's local frame. Record the particle sampling, normalization, action bounds, and continuous-to-discrete grasp-index conversion.

Reported settings to reproduce where possible:

| Setting | Paper value |
| --- | --- |
| Insertion workspace | 0.16 m × 0.16 m |
| Original clip opening | 2.2 cm |
| Initial cable-position randomization | 10 cm × 5 cm rectangle |
| Initial cable-angle randomization | −10° to +10° |
| Friction coefficient | 0.5 |
| Initial curriculum | Clip scale 1.0–2.2 for the first 1,600 reported training steps |
| Later curriculum | Clip scale 0.9–1.5 |
| Total training | 6,200 reported steps; step semantics need clarification |
| Local insertion horizon | Up to seven low-level attempts |
| Single-clip evaluation | 100 held-out scenes, original clip scale |
| Endpoint success margin | More than 2 cm beyond the clip exit, with valid insertion |

Implement the reward from Equations 1–4: partial/full insertion rewards, collision penalty, episode-length penalty, stage-specific distance shaping, and cable-front flattening. Reported coefficients are −2 for collision, −0.001 for the horizon penalty, and 0.5 for flattening. The negative horizon coefficient is not the SAC discount factor. Resolve distance signs, region boundaries, and particle-index conventions explicitly instead of silently guessing them.

The paper leaves important training details unspecified. Record network widths, learning rates, SAC discount, entropy settings, replay size, warm-up, batch size, update ratio, environment parallelism, and training seed. Count environment transitions, optimizer updates, primitive executions, simulator time, and wall time separately. Do not treat 6,200 reported steps as a known GPU-hour budget.

If training a new policy, select the checkpoint on a separate validation set and then evaluate it once on 100 held-out single-clip scenes. Report success, signed endpoint distance, collisions, and action count. The paper reports 87% success, 2.59 cm average signed distance, and 3.86-step average episode length. Reproducing the heuristic on the same 100 initial states is useful for checking task difficulty, but optional for the minimal horizon study. Do not use this held-out set repeatedly for checkpoint or hyperparameter selection.

### Implement Pull and Flatten

Pull and Flatten are scripted primitives, not additional learned policies. Specify their grasp sites, direction, distance, lift height, rotation, release, and collision handling. Both must preserve earlier clip insertions when physically possible. Flatten must perform a physical recovery rather than resetting the simulation or straightening particles directly.

Check normal use, repeated failure recovery, and the case where an inappropriate Flatten could undo an insertion or collide with a clip. Freeze these primitives before the primary evaluation.

## 6. Planner interface and information boundaries

Use an explicit ordered list of clip IDs, with consistent visible labels such as C01 through C10. This simplifies language grounding so the first experiment focuses on longer execution and progress monitoring. It is a documented change from the paper's color, spatial, and implicit instructions.

Provide the VLM with skill definitions, usage conditions, counterexamples, fixed in-context examples, the task instruction, current full-scene image, current clip crop, action history, and a consecutive-insertion-command counter. Define Insert, Pull, and Flatten clearly, including when Flatten is inappropriate after successful insertion.

For the five-target task, explicitly say that only the listed first five clips are required. The remaining clips are obstacles and are outside the requested route. Completion means finishing the requested list, not all visible clips.

Use a structured decision, for example:

```json
{
  "skill": "Insert",
  "target_clip": "C03",
  "done": false
}
```

When `done` is true, skill and target are null and no further motion is executed. When false, require a valid skill and target. Use schema-constrained final output if the serving backend supports it correctly with reasoning enabled. Otherwise allow one fixed formatting-only retry that supplies the schema and the validation error, with no geometric or semantic feedback. After that, an invalid decision is an interface failure. Log every retry and include it in query and token costs. Never silently correct a valid but wrong target, skill, or completion flag. This prevents trivial JSON formatting errors from masquerading as manipulation or progress-tracking failures.

Choose the initial crop from the first requested clip. Subsequently center it on the last commanded target. Do not choose crops using hidden knowledge of the next incomplete clip.

The initial memory policy retains the current scene and crop, fixed example images, and the complete structured action history, rather than every historical image. Set `preserve_thinking=false` explicitly if using this policy; archived reasoning can remain in the logs without being returned as history. This is a reconstruction choice, not a claim about the authors' implementation. Validate and freeze the policy on development scenes.

Keep simulator routing-success labels, true completed-clip counts, and the reference controller's next-target decisions out of the VLM prompt. Simulator state remains available to the low-level state-based policy, consistent with the architecture. Reset all conversation state between episodes.

## 7. Controllers

### VLM controller

The VLM selects the target and skill, monitors progress, decides when to recover, and chooses when to stop. Insert uses the frozen RL policy. Pull and Flatten use the frozen scripts. There is no automatic advance to the next clip based on simulator success hidden behind the VLM interface.

The command counter supplied to this controller counts repeated Insert commands to the same target; reset it on a target change or a Pull/Flatten command. This bookkeeping rule does not expose geometric success. Log the counter and use the same rule at both horizons.

### Ground-truth state reference controller

Use the commanded route and simulator geometry to track progress. Implement and freeze a deterministic state machine:

1. Select the first required clip outside the currently valid ordered prefix. If progress regresses, reconsider the earlier unsatisfied clip.
2. If partially inserted, use Pull when the geometry indicates that pulling can complete passage.
3. After a completed insertion, use Pull at the previously traversed clip when extra cable or repositioning is needed before attempting the next clip. Do not assume that a head just beyond an exit is already ready for the next insertion.
4. When the head is in the insertion region, use Insert, with the same local RL horizon as the VLM controller. If the head is in a recoverable unsuitable configuration, use Flatten according to a frozen geometric rule.
5. After three unsuccessful Insert invocations at the current target, execute Flatten and reset the retry counter.
6. Advance only on geometrically verified insertion; stop when the requested route is complete. Continue bounded recovery for unthreading or incorrect routing when the existing skill set permits it.

Define the partial-insertion, preparation-Pull, alignment, and recovery predicates on development scenes. Document precisely which clip a Pull command refers to and how its destination is determined; the same command must have the same primitive semantics for both controllers. The reference cannot move cable particles directly, reset a scene, or use an extra repair skill unavailable to the VLM.

This reference differs from the paper's greedy symbolic planner: it follows the commanded order rather than choosing the nearest clip. It provides a useful comparison with reliable state tracking, but also changes the high-level action-selection policy and creates different physical trajectories. It therefore does not isolate perception or memory by itself. It is not an optimal oracle and does not guarantee that every action or recovery is best possible.

Both controllers must use identical primitives, physical execution rules, local policy, collision checks, and horizon-scaled action allowance.

## 8. Scene design: isolate required horizon

The primary manipulation changes the required number of insertions while keeping the initial scene fixed.

The screen uses 30 held-out ten-clip scenes from the official test split ("Official data" below). Each has an explicit feasible route C01 through C10 and no initially threaded clips. For each saved initial state, evaluate the first five and all ten targets with each controller.

| Property | Five-target condition | Ten-target condition |
| --- | --- | --- |
| Physical clips present | All ten | The same ten |
| Required route | C01–C05 | C01–C10 |
| Camera, framing, resolution | Fixed | Identical |
| Clip size and opening geometry | Fixed | Identical |
| Cable length and initial state | Fixed | Identical |
| Insertion policy and scripts | Frozen | Identical |
| Planner settings and memory policy | Frozen | Identical |
| Execution allowance | 5b skill invocations | 10b skill invocations |

Keep the ten-clip route within the robot's reachable workspace. Do not shrink clip openings, reduce inter-clip spacing, or zoom out only for the ten-target condition. Balance local spacing and turn difficulty across route positions and across scenes so that the last five targets are not systematically harder by construction.

Use enough cable for the complete route and a fixed anchoring convention. Prefer a route that the original cable can reach. If a longer cable is needed, use that same cable in both conditions and revalidate the five-target task. Preserve the policy's observation dimensions and sampling semantics; do not silently change particle count or stretch the particle discretization to fit a checkpoint.

Predefine geometric feasibility checks and log rejected generation seeds before running either controller. Do not discard a scene because a controller failed on it. Use development scenes to demonstrate that the primitives can execute representative complete ten-clip routes without simulator resets.

Use a narrow, explicit route family for the initial screen: repeat locally comparable insertion/turn motifs, with bounded spacing and angle randomization. Counterbalance mirrored or reversed motif sequences so local difficulty is not tied to early or late route positions. Archive per-stage distance, relative angle, reach margin, and initial cable slack. Prohibit self-crossing routes in this first family unless they are deliberately part of its definition. Keeping the same scene across a pair controls the starting environment; it does not remove the physical constraints accumulated after five insertions. Those constraints are a legitimate mechanism to measure.

The official data uses six explicit motif families, so it is broader than a single narrow family. The screen pools the six shapes in its primary comparison. The paired design absorbs the added between-scene variation, but with five scenes per shape in the screen, per-shape results are exploratory only. Local difficulty also differs by shape: square and zigzag corners reach about 90°. Counterbalancing applies within each shape.

### Official data

The official scenes for this reproduction are the `shapes` dataset: six route shapes (arc, circle, spiral, serpentine, zigzag, square) crossed with route lengths of 3–10 physical clips, 48 cells in all.

- **Generation.** `python -m cablerouting make-dataset` builds it from the `dataset` section of `configs/default.yaml` (generator route-family-v4). Shape definitions are in the README; reconstruction choices and caveats are in deviation L6.
- **Balance.** Every split holds the same number of scenes per cell, balanced over the four mirror × reverse variants.
- **Seeds.** Seed ranges are disjoint between splits and from every other scene pool.

| File | Protocol role | Contents |
| --- | --- | --- |
| `manifests/shapes_train.json` | Development pool (Section 9) | 960 scenes, 20 per cell |
| `manifests/shapes_validation.json` | Calibration pool (Section 9) | 384 scenes, 8 per cell, generated as a separate pool |
| `manifests/shapes_test.json` | Evaluation pool, held out | 960 scenes, 20 per cell; 120 have ten clips |
| `manifests/shapes_screen.json` | The screen's 30 initial states (Section 10) | Subset of test |
| `manifests/shapes_calibration_r1.json`, `_r2.json` | Calibration rounds 1 and 2 (Section 9) | Subsets of validation; 24 five-clip and 24 ten-clip scenes each |
| `manifests/shapes_split.json` | Split record | Seeds, selection rules, scene IDs per split and subset, manifest hashes |

Subsets are fixed by rule from scene IDs and variants, never from outcomes:

- **Screen.** For the s-th shape in dataset order, take its ten-clip test scenes, lowest scene ID first within each variant. Cycle over the four variants starting from variant s mod 4, and take five. That gives 5 scenes per shape and 30 in total; the variants contribute 8, 8, 7 and 7.
- **Calibration round r.** For every shape, at both five and ten clips, take the validation scene of rank r (by scene ID) within each variant. That gives 4 scenes per shape and 24 per length.
- **Confirmation after the screen** (Section 12). Use the 90 ten-clip test scenes outside the screen, plus 24 extension scenes, 114 in total. The extension scenes are indices 40–43 of each ten-clip train/test cell, under the same seed scheme, all assigned to test, one per variant per shape. Generate them only once the confirmation is declared; generation does not depend on screen outcomes.

The primary study uses only the ten-clip test scenes and the five- and ten-clip validation scenes. The other lengths serve development, insertion-policy training, and length-based extensions (Section 13). The insertion policy's route-stage training starts sample the stages of the train split (all lengths), never validation or test geometry (deviation R8).

Keep the validation and test splits held out:
- Never tune on them, and do not render or review test layouts before the evaluation runs.
- Record any validation or test layout that is inspected outside its prescribed use in `reports/`. It counts as used and cannot become fresh held-out data again.
- If the shape families change before the configuration is frozen, regenerate every split; no outcome has been observed on any of them.
- **Settled initial states.** States exist for the screen, both calibration rounds and the train split: 1,086 states, made 2026-11-02, all passing. Each is checked against predeclared conditions before any controller runs (`scenes/state_checks.py`):
  - no clip is threaded or partially entered;
  - the head lies inside C01's insertion region;
  - the cable lies flat on the table, outside every clip's footprint;
  - settling moved no point more than 2 cm from the generated polyline.
- **Same physics or regenerate.** Each state index records a hash of the physics settings, and `freeze` refuses states made under other settings or that failed a check. If the physics is retuned before freezing, rerun `make-states --force`. Settling is bitwise reproducible regardless of process or order.
- **A state that fails a check.** The scene is logged as rejected, and the subset rule moves to the next-ranked scene of the same shape, length and variant. This happens before any episode.

This design measures task horizon at fixed initial clutter. A later five-physical-clips versus ten-physical-clips comparison would additionally measure the effects of extra clutter and scene complexity. Report that as a separate experiment.

## 9. Development, calibration, and freezing

Use three disjoint layout pools: development for tuning, calibration for the final readiness check, and evaluation for the horizon test. In this reproduction they are the train, validation, and test splits of the official data (Section 8). Use development data for simulator debugging, primitive design, prompt changes, generation settings, and action-budget selection. Record any inspected calibration or evaluation layouts as used; they cannot become fresh held-out data again.

Perform two calibration checks, using the scenes of one calibration round:

1. Five-clip validation scenes with all five requested, establishing that the reconstructed system performs credibly at the reported scale. This replaces the separate paper-like family.
2. Ten-clip validation scenes with only the first five requested, establishing a strong baseline under the clutter and camera conditions used in the test.

A proposed engineering gate is at least 20 successes in 24 fresh calibration episodes for each check **and each controller**, after tuning is complete. Each round has 24 states per check, 4 per shape, so the gate keeps the original 16/20 threshold of at least 80% while covering every shape equally. Pair the calibration states between controllers. One round requires 96 calibration episodes in addition to the 120-episode screen. The 80% threshold is our screening choice, not a paper result or a statistically established population success rate.

Establish ten-clip physical feasibility on designated ten-clip train scenes, with the reference or a manually specified sequence of the shared skills. Do not require the VLM to succeed at ten clips as a readiness gate: that would select away the limitation being investigated. Do not use any controller's ten-clip outcome to select evaluation scenes.

If a five-target gate fails, return to development, retain the failed calibration report, and use the next calibration round's states for the next version. If both rounds are used up, generate further validation scenes, under the validation seed scheme, before the next attempt. A weak reference cannot serve as evidence that failures shared by both controllers are purely physical.

Select the per-clip budget `b` using development episodes before the fresh calibration checks. Start with ten high-level skill invocations per requested clip and increase it if normal successful development runs are budget-limited. Keep the local insertion horizon and recovery rules fixed. Document the final choice before evaluating held-out scenes.

Audit the observed five-target success rate again on the evaluation data. A high development or calibration rate does not guarantee a high held-out rate. If the evaluated five-target rate is low, report the comparison but limit claims about a new breakdown emerging specifically beyond the paper's horizon.

Freeze the scene generator, initial-state manifest, all assets and physics settings, policy checkpoint, primitive code, evaluator, prompts, example images, model revision, sampling profile, context/history policy, and budgets. Record their hashes or immutable revisions.

The 120 primary episodes exclude training, single-clip checks, and calibration. A one-seed insertion policy is sufficient for this minimal experiment, but conclusions are conditional on that checkpoint. Multiple independent training seeds are a later robustness extension.

## 10. Primary evaluation procedure

Run the following four conditions on each of the 30 saved initial states of `manifests/shapes_screen.json`:

| Controller | Required clips | Episodes |
| --- | ---: | ---: |
| VLM | 5 | 30 |
| VLM | 10 | 30 |
| Ground-truth state reference | 5 | 30 |
| Ground-truth state reference | 10 | 30 |
| **Total** | | **120** |

Randomize condition order within each scene and record the scheduling seed. Pair the same initial-state and policy RNG seeds across conditions where supported. Different action sequences will cause trajectories and RNG consumption to diverge; the pairing guarantees common initial conditions, not identical stochastic futures.

For each episode:

1. Load the saved initial state and reset controller memory, counters, and RNGs.
2. Confirm that the scene, camera, cable, and requested-route manifest match the condition.
3. Obtain a controller decision, execute the selected skill, and record geometric progress and any failure events.
4. Continue until the controller declares completion, a terminal collision or unrecoverable execution error occurs, or the execution budget is exhausted. Recoverable mistakes remain in the episode.
5. Grade the final state independently and archive the complete episode trace.

Set the execution budget to `B(N) = bN`; with `b = 10`, this is 50 versus 100 high-level skill invocations. Count internal RL attempts, physical pick-and-place operations, and model queries separately. The paper's episode-length metric counts pick-and-place steps, so high-level invocations should not be presented as directly interchangeable with it.

Allow a completion query after the last permitted physical skill invocation; a successful final insertion must not fail solely because there was no opportunity to declare completion. No further physical action may exceed the budget. A ten-target run can spend more of its total budget on the first five targets than a five-target run can. Therefore also report prefix attainment within a common budget of `bk` for each prefix k; use that measure when comparing early progress across horizons. Keep ordinary total-budget success as the primary endpoint.

Scale any task-execution time cap with horizon as well. Prefer a skill-count primary budget so inference latency does not alter the task difficulty. Record real wall time as an efficiency metric. Do not retain a five-clip timeout unchanged at ten clips.

Terminal robot/gripper collisions end an episode under the frozen collision rule. Wrong-target threading and loss of an earlier insertion are recorded as errors, but are not automatically terminal: allow the controller to attempt repair with its existing skills and remaining budget. Correct final routing means the anchored-end-to-head cable path traverses exactly the requested clips in the requested directions and order, with all required threading still present. Record chronological insertion-order violations separately, including violations later repaired. This operational definition must be documented because the paper does not completely specify repaired mistakes.

A VLM `done` decision before true completion is a premature-termination failure. A successful final state must pass the independent routing check. Do not abort at the first recoverable error: the architecture's ability to recover is part of what is being tested, and disabling it would artificially compound failures at longer horizons. A strict first-error failure score may be computed from the same traces as a secondary diagnostic, without stopping the primary rollout.

Keep model-format errors, generation cutoffs, context overflow, infrastructure failures, and physical task failures distinguishable. Do not silently remove failed runs. For a crash or service outage that makes an episode unscorable, retain the failed attempt, document the cause, and apply a declared rerun rule independent of task outcome. If a configuration change is needed, version it and rerun the affected comparison consistently.

Do not adapt prompts, checkpoints, or budgets after inspecting held-out outcomes. Do not stop when a desired number of failures has appeared. The progress-information diagnostic in Section 12 is a separate condition, not a replacement for failed primary runs. Seven-clip, larger-sample, or additional-seed runs form separately declared extensions.

## 11. Logging and metrics

For every episode save the scene/seed, controller, horizon, versions and hashes, requested route, initial state, final state, and terminal reason. For each decision save images, crop bounds, prompt/history, model response, structured action, token counts and finish reason, skill trajectory, counters, geometric insertion state, collisions, and timing. Archive reasoning only when the serving interface provides it; the evaluator does not depend on receiving reasoning text.

Use these primary and diagnostic metrics:

| Metric | Definition |
| --- | --- |
| Full-task success | Correct final ordered routing, including any repaired mistakes, plus a correct completion decision within the budget and no terminal failure. |
| Five-to-ten success decrease | Success at five targets minus success at ten targets, separately for each controller. |
| Progress survival `S(k)` | Fraction of runs that reach the first k correct consecutive insertions before terminal failure. |
| Conditional stage failure `h(k)` | Among runs that reached k−1 insertions, fraction that fail to reach k. |
| Attainment within equal prefix budgets | Fraction reaching the first k correct insertions within `bk` physical skill invocations, for comparisons across horizons. |
| Final valid prefix length | Number of correctly threaded consecutive required clips at termination. |
| Efficiency | Skill invocations, internal insertion attempts, physical operations, recovery calls, model tokens, and wall time. |

Track the currently valid ordered prefix after every action, its greatest value ever reached, all regressions, and whether each regression was repaired. Define `S(k)` using first attainment of a simultaneously valid prefix of length k; do not combine insertions that were never valid together. Set `S(0) = 1`. Its first-attainment failure rate is `h(k) = 1 - S(k)/S(k-1)` where `S(k-1) > 0`. This diagnostic does not measure retention of earlier insertions or difficulty after a regression. Report regression and repair rates separately, and mark an empty at-risk set as undefined.

A run can reach every requested clip and then fail through an unnecessary action or a failure to terminate correctly. Record these post-completion failures separately: progress survival at the final clip need not equal full-task success.

Label observable failure events as wrong-target/order decisions, premature completion, unnecessary repeat insertion, inappropriate recovery, unrecovered local insertion failure, collision, unthreading, or budget exhaustion. Keep infrastructure and output-format failures in separate categories. Assign a terminal cause and allow multiple contributing events. Review all failures rather than selecting only illustrative ones.

## 12. Analysis and interpretation

Prespecify a 20-percentage-point decrease as the effect of practical interest for this initial design. This is a study-design choice, not a prediction. The primary comparison is the VLM's five-versus-ten full-task success. Reference comparisons, stage curves, and failure categories are diagnostic; do not select the most favorable metric afterward as the primary finding.

Report each success rate with numerator/denominator and a 95% Wilson interval. For the paired difference, report the full 2×2 outcome table, particularly `n10` (five succeeds, ten fails) and `n01` (five fails, ten succeeds). Use a two-sided exact McNemar test at alpha 0.05 for the primary null of equal marginal success; a positive estimate is additionally necessary to claim a decrease. [Exact McNemar test][mcnemar]

Use a confidence interval that respects pairing and remains informative at the boundaries. One conservative, fully specified option is to calculate two 97.5% two-sided Clopper–Pearson intervals for the marginal discordance probabilities `p10` and `p01`, each using all n scene pairs, then report `[L10 - U01, U10 - L01]` for their difference. Bonferroni coverage is at least 95%; this interval can be wider than a paired score interval or the exact test's implied acceptance region. State which method was used. Never report a zero-width bootstrap interval from identical observed outcomes as evidence of exact equality.

For descriptive uncertainty on the difference in controller decreases below, a paired scene bootstrap with 10,000 resamples and a recorded seed is useful: resample all four outcomes for a scene together. Mark bootstrap boundary/degeneracy cases explicitly. If later adding repeated runs per scene, preserve the scene clustering and revisit the testing procedure; repeated runs are not independent new layouts.

Define the descriptive difference in decreases as:

```text
Delta_VLM = SR_VLM(5) - SR_VLM(10)
Delta_REF = SR_REF(5) - SR_REF(10)
I = Delta_VLM - Delta_REF
```

A positive `I` indicates a larger decrease for the VLM under this comparison. Different high-level decisions produce different rope configurations, so this contrast does not identify perception, memory, or reasoning as the cause by itself. Report absolute rates as well as decreases: ceiling/floor effects can make `I` misleading.

### Sample size and decisions

Thirty paired scenes can easily miss a meaningful decrease. The following exact-test power calculations assume a true 20-point decrease and independent scene pairs; `d` is the probability that a pair has different outcomes at the two horizons. They sum a two-sided exact McNemar rejection probability over `M ~ Binomial(n, d)` discordant pairs, with `n10 | M ~ Binomial(M, (d + 0.20)/(2d))`. The values are design calculations, not measured task performance.

| Paired scenes | Core episodes | Power if d = 0.30 | Power if d = 0.50 |
| ---: | ---: | ---: | ---: |
| 30 | 120 | 40% | 26% |
| 60 | 240 | 80% | 52% |
| 100 | 400 | 96% | 79% |
| 114 | 456 | 98% | 84% |

Choose the study size before examining its evaluation outcomes. Retain 30 scenes for a minimal screen. If a reasonably sensitive confirmatory comparison is the immediate goal, use 100 scenes from the outset. A separate confirmation after the screen uses at least 100 fresh scenes, a fixed sample size, and separately reported results; do not pool selectively or extend sampling until significance.

This reproduction runs the screen first (decided 2026-11-02, before any evaluation episode). Its confirmation, if declared, uses the 114 fresh ten-clip test scenes fixed in Section 8.

Interpret uncertainty explicitly: a positive decrease with the prespecified exact test rejecting equality supports deterioration in this implementation. A lower confidence bound above 0.20 supports a decrease larger than the practical threshold; an upper bound below 0.20 rules out that size under the tested conditions. Otherwise the practical magnitude remains unresolved. A nonsignificant 30-scene screen does not establish robustness.

Produce three core figures:

1. Full-task success for the four conditions, with uncertainty intervals.
2. Progress-survival curves through clips 1–10, with five-target curves ending at five. Do not treat successful five-target episodes as failures after their requested endpoint.
3. Conditional first-attainment failure by stage, with the number of runs reaching each stage, plus a breakdown of terminal failure mechanisms and progress regressions/repairs.

Late-stage failure estimates are based on surviving runs and can be noisy or affected by survivor selection. Interpret them alongside balanced local geometry and the reference controller rather than treating an increasing curve as automatic proof of memory failure.

### Separate compounding risk from worsening stage reliability

Under a simplified stationary, independent-stage model, effective post-recovery reliability `p` gives `P(complete N) = p^N`. For illustration, `p = 0.95` gives approximately 77% completion at five clips and 60% at ten clips. This is an explanatory model, not an expected result from the paper.

Estimate effective reliability from completed clip stages and recovery behavior in our data. Do not substitute the paper's 87% single-insertion success directly into this formula: retries, pulling, recovery, geometry, and accumulated constraints change the effective probability.

Do not treat a gap below `SR(5)^2` as proof of an extra horizon effect. Scene difficulty, dependent errors, retained threading, final stopping decisions, and the global action budget violate the simple independent-stage assumptions. Use the formula as a reference explanation. Stronger mechanism claims require observed failure traces and the intervention below.

| Observation | Interpretation supported by the experiment |
| --- | --- |
| Both controllers decline with roughly stable conditional stage failure | Accumulated execution risk is a plausible explanation, not an identified cause. |
| Both show higher failure at later stages | Accumulating constraints, geometry, or skill limitations are plausible; inspect physical traces. |
| VLM declines more than the state reference | High-level policy differences plausibly contribute; use the progress-information intervention before identifying monitoring as the cause. |
| VLM increasingly stops early, repeats finished actions, or selects wrong targets | Direct evidence of specific progress-tracking or decision errors. |
| VLM declines mainly through inappropriate Flatten actions | Recovery decisions or recovery interactions contribute; distinguish them from failed Insert execution. |
| Ten-target degradation is dominated by context/output cutoffs | The tested serving configuration imposes a limit; separate it from evidence about the architecture. |
| No significant decline is detected | Report the estimate, interval, and sample-size sensitivity; an imprecise screen is inconclusive. |

An observed completion decrease is a practical scaling limitation even with constant stage reliability. A larger VLM/reference gap accompanied by identifiable decision errors supports the stronger conclusion that the VLM hierarchy contributes to the horizon problem. Generalization beyond this model, scene family, and insertion checkpoint requires additional evidence.

### Progress-information diagnostic

If the screen is used to claim a progress-monitoring limitation, run a separately logged **VLM + true progress** condition on all 30 ten-target scenes, adding 30 episodes. Prespecify this diagnostic before the screen and run it on original successes as well as failures. Use the same model, seed schedule, images, skills, history policy, and budgets. Add only a compact simulator-derived summary of the currently valid ordered prefix, the first unsatisfied requested target, and whether all requested threading is currently valid. Do not prescribe the skill or add a new recovery primitive. This is privileged information and must be labeled as such.

Compare this condition with the original ten-target VLM using paired outcomes and error traces. A substantial improvement supports sensitivity to reliable task-state information. It does not separate visual perception from remembering prior progress, and it does not prove that longer context alone is responsible. Little improvement directs attention toward skill execution, recovery choice, geometric constraints, or other planner limitations. If those mechanisms remain ambiguous, report them as unresolved rather than choosing one from the success-rate curve.

Use explicit operational labels when reviewing traces: for example, `done=true` with an incomplete geometric route, a repeated Insert after stable full passage, or Flatten followed by loss of valid threading. These observations are stronger than attributing an internal mental cause from fluent reasoning text. Review cases using a frozen labeling guide, and record uncertain or multiple causes.

## 13. Optional extensions

Only add these after the primary protocol is frozen and completed:

- A seven-clip condition to examine the shape of the horizon trend.
- A second capable open-weight model family, using the same physical experiment and development-only prompt adaptation.
- More specific interventions after the true-progress diagnostic, such as improved images versus a persistent progress record, to distinguish perception from retaining state across decisions.
- A five-versus-ten physical-clip comparison to measure additional clutter, separately from the fixed-scene horizon test. The five- and ten-clip test scenes of the official data share shapes and seeding, so they support it directly.
- Multiple VLM sampling seeds and independently trained insertion policies.
- Harder routes with separately controlled spacing, turns, cable length, and topology. Avoid bundling these changes into the primary horizon manipulation.

For a broader reproduction of Table II, evaluate the five original controller combinations across implicit, spatial, attribute, and combined 4/5-clip settings at the stated 15 trials per category: approximately 300 episodes. Implement the original greedy symbolic planner separately from our commanded-order reference. Reproduce the no-recovery ablation as described, while acknowledging that it also removes Flatten. This broader benchmark is outside the minimal 120-episode horizon test.

## 14. Optional physical reproduction

Physical replication is outside the initial experiment. The paper uses a Franka Emika Panda, calibrated wrist-mounted RealSense D415 with 1280 × 720 top-down images, ROS/MoveIt, and SAM2 segmentation on an RTX 5090. Cable/clip state is extracted from segmentation for the low-level policy; reference-point initialization is part of the perception procedure. The paper reports no additional real-world policy fine-tuning. [Paper][paper]

To reproduce that stage, implement calibration, image-to-state extraction, the same primitive execution interface, and independent outcome annotation. Evaluate the original eight configurations against fixed-order + RL before extending the horizon. New robot hardware, a different perception system, or ten-clip layouts introduce additional transfer conditions and should be reported explicitly.

## 15. Deliverables, effort, and completion criteria

Maintain an implementation layout along these lines; these paths are proposed artifacts, not files already supplied by this protocol:

```text
configs/             pinned environment, policy, planner, and evaluation settings
assets/              cable/clip scenes and provenance
prompts/             skill definitions, examples, output schema, and history policy
checkpoints/         frozen insertion checkpoint and training metadata
manifests/           official train/validation/test splits, screen and calibration subsets, seeds, rejected-layout log
runs/                episode traces, images, actions, geometry, and run manifests
analysis/            metrics, paired comparisons, and plots
reports/             calibration results, deviation log, and final findings
```

The main engineering dependency is a credible manipulation environment and insertion policy. A paper-only simulation reconstruction was estimated in discussion at roughly 6–12 weeks for an experienced robotics/RL engineer; a working author release could reduce that to roughly 1–3 weeks. Physical reproduction could add 4–8 weeks with an existing lab. These are planning estimates, not measured durations. Instrument training and inference before estimating GPU-hours. The 120-episode screen includes 60 VLM-controlled episodes; calibration, the progress diagnostic, and confirmation incur additional local inference. No GPT-5 API spending is required.

| Study component | Episodes, excluding training and open-ended development |
| --- | ---: |
| Fresh calibration: two checks × two controllers × 24 states, per round | 96 |
| Minimal horizon screen: 30 states × four conditions | 120 |
| True-progress diagnostic on the screen's ten-target states | 30 additional |
| Confirmatory horizon study after the screen: 114 fresh states × four conditions | 456 |
| Single-clip evaluation for a newly trained insertion checkpoint | 100 additional; another 100 if evaluating the heuristic |

Thus the calibrated minimal screen requires 216 closed-loop routing episodes, or 246 with the progress diagnostic, plus any insertion evaluation and any repeated calibration round. The 120-episode figure refers only to the four-condition screen.

The experiment is complete when:

- [ ] Assets, environment, evaluator, and resets have been validated.
- [ ] The insertion checkpoint and scripted skills are frozen and their calibration results recorded.
- [ ] Both controllers passed fresh five-target calibration checks; perception, serving/context, and ten-clip physical-feasibility checks are documented.
- [ ] Reconstruction choices, model settings, action budgets, and analysis rules are frozen.
- [ ] The chosen sample size was frozen, and all 30 screening or 114 confirmation scene IDs have all four primary outcomes, with every failed attempt and rerun explained.
- [ ] Success, progress survival, conditional failure, efficiency, and failure mechanisms are reported.
- [ ] Paired uncertainty estimates and representative traces accompany the interpretation.
- [ ] Any progress-monitoring claim is supported by the prespecified true-progress diagnostic, with alternative causes and remaining uncertainty reported.
- [ ] The report distinguishes measured results, reconstruction gaps, and limits on claims about the original paper.

## Sources

The paper-specific details above are drawn from the supplied v2 PDF and its versioned public text. Software and model availability were checked on the protocol date. Pin local copies and immutable revisions when preparing the experiment.

- [Paper v2, method, evaluation, and limitations][paper]
- [Authors' project page][project]
- [Authors' public website repository][authors-code]
- [GarmentLab implementation and installation instructions][garmentlab]
- [Qwen3.8-27B model card and serving documentation][qwen]
- [Qwen3.5-9B optional development model][qwen-small]
- [vLLM Qwen3.8-27B recipe and verification scope][vllm]
- [NVIDIA L40S specifications][l40s]
- [Exact paired binary-outcome test: McNemar][mcnemar]

[paper]: https://arxiv.org/html/2510.19268v2
[project]: https://icra2026-dloroute.github.io/DLORoute/
[authors-code]: https://github.com/icra2026-dloroute/DLORoute
[garmentlab]: https://github.com/GarmentLab/GarmentLab
[qwen]: https://huggingface.co/Qwen/Qwen3.8-27B
[qwen-small]: https://huggingface.co/Qwen/Qwen3.5-9B
[vllm]: https://recipes.vllm.ai/Qwen/Qwen3.8-27B
[l40s]: https://www.nvidia.com/en-us/data-center/l40s/
[mcnemar]: https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html
