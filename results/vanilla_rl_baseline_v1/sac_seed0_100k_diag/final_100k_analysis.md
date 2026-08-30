# SAC seed 0 — 100k final analysis

## Completion

- Status: `COMPLETED`, exit code 0
- Steps: 100,000
- Runtime: 23,681.13 seconds (6 h 34 m 41 s)
- Throughput: 4.223 environment steps/s
- Replay-buffer size: 100,000
- Actor, critic, and entropy parameters remained finite
- Final entropy coefficient: 0.006607

## Fixed-seed final comparison (100 episodes per model)

| Metric | Best checkpoint | Last / 100k checkpoint |
|---|---:|---:|
| Success rate | 10% | 5% |
| Mean return | 71.299 | 60.968 |
| Return per step | 0.3801 | 0.3182 |
| Grasp rate | 79% | 57% |
| Mean grasp duration | 92.98 steps | 65.45 steps |
| Mean maximum object height | 0.832476 m | 0.831655 m |
| Mean maximum lift delta | 1.430 mm | 0.609 mm |

The best checkpoint is the checkpoint selected by the online evaluation rule
(success rate first, mean return second). Its modification time and the online
curve indicate that it was selected around the 70k evaluation, where the
20-episode online success estimate was 5%.

## Online evaluation trend

| Step | Success | Return | Grasp | Grasp duration | Mean lift delta |
|---:|---:|---:|---:|---:|---:|
| 10k | 0% | 22.47 | 0% | 0.00 | 0.000 mm |
| 20k | 0% | 32.51 | 40% | 13.65 | 0.164 mm |
| 30k | 0% | 69.96 | 0% | 0.00 | 0.000 mm |
| 40k | 0% | 68.53 | 25% | 9.30 | 0.056 mm |
| 50k | 0% | 69.19 | 65% | 61.85 | 0.737 mm |
| 60k | 0% | 76.06 | 60% | 67.30 | 0.084 mm |
| 70k | 5% | 75.34 | 65% | 94.10 | 0.663 mm |
| 80k | 0% | 80.73 | 95% | 111.75 | 0.159 mm |
| 90k | 0% | 80.95 | 70% | 90.80 | 0.079 mm |
| 100k | 0% | 60.55 | 55% | 66.40 | 0.000 mm |

Each online point uses only 20 episodes, so zero at an individual checkpoint
does not contradict the 5–10% rates measured by the final 100-episode test.

## Interpretation

This is a positive but weak vanilla baseline. SAC learned reaching and frequent
grasping, and genuine successes are reproducible, so the environment and
learning pipeline are not fundamentally broken. It did not learn a stable lift
policy: lift deltas remained far below the approximately 9 mm additional height
usually needed to cross the local 0.84 m success threshold, and the 100k policy
regressed relative to the best checkpoint.

The return curve is not a reliable proxy for task completion. At 80–90k the
online return was highest while success remained zero, consistent with long
episodes accumulating reaching and grasp reward without completing the lift.

## Decision

- Use `best_model.zip`, not `last_model.zip`, as the reported seed-0 policy.
- Report vanilla SAC seed-0 success as **10/100 for best checkpoint** and
  **5/100 for the final checkpoint**.
- Do not describe 100k as converged.
- Before spending substantially more samples, run the already proposed minimal
  fixed-horizon-semantics A/B. Keep all other SAC settings fixed.
