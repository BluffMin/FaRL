# Signed lift-progress reward A/B — final report

## Executive conclusion

The treatment produced a small, statistically uncertain best-checkpoint improvement but did not solve the grasp-to-lift bottleneck and regressed by 300k. Verdict: `LIFT_REWARD_EFFECT = NOT_SUPPORTED_WITH_THIS_SHAPING`.

## Final matched 100-episode comparison

| Metric | Control best | Treatment best | Control last | Treatment last |
|---|---:|---:|---:|---:|
| Success | 0% | 5% | 0% | 1% |
| Grasp | 100% | 89% | 95% | 100% |
| Lift | 2% | 5% | 0% | 2% |
| P(lift given grasp) | 2% | 4.49% | 0% | 2% |
| Mean lift delta | 0.310 mm | 0.818 mm | 0.022 mm | 0.294 mm |
| Mean total return | 88.53 | 77.67 | 88.60 | 92.62 |

The paired best-model success difference was +5 percentage points, with bootstrap 95% CI [0.01, 0.1]. The lift-rate difference was +3 points, CI [-0.02, 0.08]; uncertainty includes no benefit. This is a single-seed causal screen.

## Learning and stability

Treatment first reached 5% online success at 140k and had its selected best around 170k (5% success, 10% lift in 20 episodes). Later successes reappeared at 280k and 290k, but the 300k online point returned to zero. Final last-model success was only 1/100. Pattern: improved but regressed / intermittent, not stable improvement.

## Mechanism

Treatment best increased physical lift modestly, but mean post-grasp z action remained strongly negative. Control best E[a_z|grasp] was about -0.865; treatment best was about -0.842. The reward did not teach coherent upward commands. Most episodes accumulated zero progress reward; mean treatment-best progress return was only 0.043.

## Exploit audit

No reward exploit was detected. Static grasp and lift-drop oscillation yield zero offline shaping, regrasp at fixed height yields zero, and final policies did not accumulate substantial progress reward without physical displacement. The failure is lack of a strong behavioral effect, not exploitation.

## Decision

Do not increase training budget again, tune entropy, or enlarge the network. This exact transition reward is not supported as a sufficient fix. The clean next causal test is the already prepared immediate-success termination versus fixed-horizon comparison, or a targeted controller/action credit-assignment test.
