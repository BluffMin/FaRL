# Artifact index

| Item | Path | Purpose / verdict | Key result |
|---|---|---|---|
| 20k sanity | `results/vanilla_rl_baseline_v1/sac_20k_sanity_tqdm/` | SAC numerical pipeline | `summary.json`; PASS_NUMERIC_ONLY |
| 100k control | `results/vanilla_rl_baseline_v1/sac_seed0_100k_diag/` | Original success-termination baseline | `final_model_comparison.json`; best 10%, last 5% |
| 300k control | `results/vanilla_rl_baseline_v1/sac_seed0_300k_from_last/` | Budget continuation | `final_300k_analysis.md`; PLATEAU_AT_GRASP |
| Forensics | `results/vanilla_rl_baseline_v1/live_training_forensic_audit_v1/` | interface/numerical audit | `final_report.md` |
| Reward A/B | `results/vanilla_rl_baseline_v1/reward_ab_v1/` | signed lift-progress | `final_summary.json`; NOT_SUPPORTED |
| Fixed 100k parent | `results/vanilla_rl_baseline_v1/sac_seed0_100k_fixed_horizon/` | fresh fixed-horizon parent | `run_status.json`, checkpoints/replay |
| Fixed 300k | `results/vanilla_rl_baseline_v1/sac_seed0_300k_fixed_horizon_from_100k/` | successful true continuation | `final_model_comparison.json`; best 92%, last 88% |
| Termination A/B | `results/vanilla_rl_baseline_v1/termination_ab_v2/` | causal screen | `final_report.md`; FIXED_HORIZON_SUPPORTED |
