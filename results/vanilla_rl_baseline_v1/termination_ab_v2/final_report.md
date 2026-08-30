# Vanilla SAC termination A/B v2 — final 300k report

## Result

Fixed horizon clearly solved the seed-0 grasp→lift bottleneck. On 100 matched final seeds, best checkpoint success increased from **0% to 92%**, lift from **2% to 92%**, and P(lift|grasp) from **2% to 98.9%**. The last checkpoint remained strong: success 88%, lift 90%.

Paired success difference was **+92%** (95% bootstrap CI 86%–97%); lift difference was **+90%** (CI 84%–95%).

## Mechanism

Post-grasp z action became materially less negative: -0.865 → -0.587, while the episode-weighted positive-z fraction rose from 0.394% to 17.6%. Fixed best mean maximum lift delta was 0.724 m versus 0.000310 m. Replay reward==1 occupancy proxy rose from 0.0163% to 23.88%.

Fixed best ever-success was 92%, final-success 74%; aggregate retention ratio was 80.4%. Continued post-success behavior is therefore useful overall but not perfectly stable.

## Interpretation

`FIXED_HORIZON_SUPPORTED`. This is a strong single-seed causal screen, not yet a multi-seed superiority claim. The operational treatment changes both terminal bootstrapping and exposure to repeated successful states/reward, so those mechanisms are not separated here. The next experiment is an exact seeds 1–2 replication without reward shaping or hyperparameter changes.
