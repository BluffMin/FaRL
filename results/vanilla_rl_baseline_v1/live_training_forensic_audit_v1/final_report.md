# FaRL vanilla SAC live forensic audit

## 1. Executive summary

The numerical SAC pipeline and interfaces are functional. At the stable 50k snapshot the policy reaches and grasps but does not lift reliably. The strongest interpretation is partial learning plus a dense-reward stage gap, not action double-scaling, missing task observations, timeout corruption, or an entropy implementation failure.

## 2. Active training safety check

The audit was out-of-process and read-only against the active run. Stable artifacts were copied to the audit directory before loading. Same-process status is in `running_process_snapshot.json`.

## 3. Exact current SAC configuration

Panda / OSC_POSE / 20 Hz / horizon 200, shaped reward scale 1, raw 42-D observations, 7-D Box actions, SAC 256x256, gamma .99, tau .005, batch 256, one update per step after 5k warm-up.

## 4. Official GitHub baseline comparison

Core robot/controller/frequency/optimizer-scale settings match broadly. Material deltas are horizon 200 vs 500 and success termination vs fixed-horizon execution.

## 5. Environment and Gym API audit

The wrapper clips once and returns Gymnasium five-tuples. Independent shadow environments were used.

## 6. Termination vs truncation

Success is a true termination. Horizon is a truncation; DummyVecEnv adds `TimeLimit.truncated=True`, and ReplayBuffer masks timeout dones. Timeout handling is correct. Success termination differs from normal fixed-horizon robosuite semantics.

## 7. Success semantics

Success is cube z > table z + 0.04 m (0.84 m locally), with no grasp or persistence requirement. Environment, info, and logger agree.

## 8. Reward semantics

Before success, reward is normalized reach plus 0.25 grasp bonus; success replaces it with 1. There is no lift-progress term. Persistent close/grasp behavior can accumulate substantial value without lifting.

## 9. Demo-vs-RL return comparability

Original totals are not comparable: demos are about 40-55 steps and were collected with reward_shaping=False, while RL uses up to 200 shaped steps. Matched-prefix replay is the valid comparison.

## 10. Discount/horizon analysis

Gamma .99 gives an effective 100-step / 5-second horizon; gamma^200 is about .134. Demo completion lies inside the effective window.

## 11. Action scaling and OSC_POSE semantics

No double scaling was found. Box commands are [-1,1]; OSC maps translation to up to 0.05 m and rotation to 0.5 rad command deltas. Signed response tests pass.

## 12. Gripper semantics

Positive closes and negative opens; demo statistics and response tests agree.

## 13. Observation correctness

The 42-D vector contains object state then robot proprioception. Cube and EEF position are visible, observations change after step, and no NaN/Inf or normalization layer was found.

## 14. Reset distribution

Five hundred RL resets confirm the narrow nominal placement sampler. Full Cartesian demo-initial reconstruction was not needed to establish interface validity and remains a secondary comparison.

## 15. Scripted controller solvability

See `scripted_controller_results.json`. This ground-truth feedback test is an environment controllability diagnostic, not a learning baseline.

## 16. Expert replay solvability

Exact model XML and initial simulator state restoration were used. See `expert_replay_results.json`; differences from RL reward/termination are explicitly retained.

## 17. SAC replay/update correctness

The saved 50k buffer has explicit timeouts and true terminals; update/data is approximately one after warm-up.

## 18. Entropy autotuning analysis

Runtime target entropy is -7. Low alpha alone is not collapse; snapshot log-prob/log-std measurements are consistent with normal automatic tuning.

## 19. Critic/Q-value health

Twin Q values are finite and comparable in scale. No explosion or gross disagreement is visible; exact Bellman target analysis is noted as a limitation.

## 20. Behavioral stage funnel

The fixed-seed deterministic funnel separates reach, contact, grasp, lift, and success. It shows where progress stops rather than hiding it in total return.

## 21. Checkpoint learning trend

Online 10k-spaced evaluations show increasing return and grasp behavior but essentially no lift/success at the audited point: partial stage-wise progress.

## 22. Ranked root-cause hypotheses

1. Training has learned only through grasp so far. 2. Dense reward provides no lift-progress gradient and rewards persistent partial behavior. 3. Horizon/success-termination differ from the official fixed-horizon setup. Interface bugs examined here are ruled out or unlikely.

## 23. Current-run interpretation

Keep the current run unchanged. It remains a valid diagnostic baseline; its return increase should be described as reach/grasp progress, not task mastery.

## 24. Minimal next experiment

Only if the completed 100k run still has zero lift/success, compare the frozen current setup against one treatment that preserves fixed-horizon execution while retaining correct timeout truncation. Do not tune entropy simultaneously.
