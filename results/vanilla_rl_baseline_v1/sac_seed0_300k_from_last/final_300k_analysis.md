# Vanilla SAC 300k continuation — final report

## Executive conclusion

Continuing the exact 100k last checkpoint and its 100k replay buffer for 200k additional transitions did **not** produce stable lift learning. The result is `PLATEAU_AT_GRASP`, not evidence that 100k was merely too early.

## Continuation validity

This was a true continuation: model, critics/targets, optimizer states, learned entropy coefficient, timestep counter, and replay buffer were restored. Effective steps advanced from 100,000 to 300,000 with 200,000 new transitions and 200,000 gradient updates. Reward, success termination, observation/action semantics, architecture, entropy configuration, controller, horizon, and seed were unchanged.

## Final 100-episode evaluation

| Metric | Parent best (~70k) | Parent last (100k) | 300k-run best | Last (300k) |
|---|---:|---:|---:|---:|
| Success | 10% | 5% | **0%** | **0%** |
| Grasp | 79% | 57% | **100%** | **95%** |
| Measurable lift | not recorded | not recorded | **2%** | **0%** |
| P(lift \| grasp) | not recorded | not recorded | **2%** | **0%** |
| Mean lift delta | 1.43 mm | 0.61 mm | **0.31 mm** | **0.022 mm** |
| Mean shaped return | 71.30 | 60.97 | **88.53** | **88.60** |

## Learning trajectory

The 20-episode online evaluations briefly reached 5% success at 170k, 180k, 230k, and 240k, then returned to zero from 250k through 300k. Meanwhile grasp rose to 100% and shaped return approached 90. This is intermittent rare lift followed by regression, superimposed on a strong grasp plateau.

## Grasp-to-lift diagnosis

At final evaluation, nearly every episode grasped, but almost none lifted by 4 mm. The best snapshot lifted in 2/100 episodes and succeeded in 0/100; the last snapshot lifted in 0/100. Thus the dominant failure is specifically `grasp -> lift`, not reach or grasp acquisition.

## Reward interpretation

The policy improved the behavior rewarded densely—approach and prolonged grasp—while physical lift and success deteriorated relative to the 100k reference. High return is therefore not evidence of task mastery. This strengthens the prior finding that the current dense reward lacks a useful continuous lift-progress incentive.

## Numerical health

Training completed normally at 300k with finite actor/critic/entropy values, replay size 300k, and alpha 0.004941. There is no evidence here that entropy tuning, architecture size, or numerical instability is the primary cause.

## Scientific verdict

`TRAINING_BUDGET_EFFECT = PLATEAU_AT_GRASP`. This is a single-seed causal screen, not population-level benchmark evidence. Nevertheless, simply extending this exact seed/configuration toward 500k is not supported by the observed trajectory.

## Recommended next experiment

Run one minimal reward-structure A/B that adds a physically scaled continuous lift-progress signal while holding termination semantics, SAC hyperparameters, architecture, entropy, observations, actions, seed, controller, and horizon fixed. Do not perform a broad sweep.
