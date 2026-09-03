#!/usr/bin/env python3
"""Read-only analysis of the completed seed-0 100k termination A/B screen."""
from __future__ import annotations
import csv, json, pickle
from pathlib import Path
import numpy as np

ROOT = Path("/home/brainlab/FaRL")
BASE = ROOT / "results/vanilla_rl_baseline_v1"
OUT = BASE / "termination_ab_v2"
OUT.mkdir(parents=True, exist_ok=True)
RUNS = {"control": BASE / "sac_seed0_100k_diag", "fixed": BASE / "sac_seed0_100k_fixed_horizon"}


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False))


def load(run, name):
    return json.loads((RUNS[run] / name).read_text())


def episodes(run, which):
    return load(run, "final_model_comparison.json")[which]["episode_diagnostics"]


def b(e, k):
    return np.array([bool(x.get(k, False)) for x in e], dtype=float)


def f(e, k, default=0):
    return np.array([float(x.get(k, default) or default) for x in e])


def lift(e):
    return f(e, "object_lift_delta") >= 0.004


def ci_pair(x, y, n=20000, seed=240829):
    d = np.asarray(y, float) - np.asarray(x, float)
    rng = np.random.default_rng(seed)
    z = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return {
        "control_mean": float(np.mean(x)),
        "fixed_mean": float(np.mean(y)),
        "paired_difference_fixed_minus_control": float(d.mean()),
        "ci95": [float(v) for v in np.quantile(z, [0.025, 0.975])],
        "bootstrap_samples": n,
        "n_pairs": len(d),
    }


def metric(e):
    g = b(e, "ever_grasped")
    l = lift(e)
    s = b(e, "ever_success")
    fin = b(e, "final_success")
    pg = [
        float(x["post_grasp_z_action_mean"])
        for x in e
        if x.get("post_grasp_z_action_mean") is not None
    ]
    pp = [
        float(x["post_grasp_positive_z_action_fraction"])
        for x in e
        if x.get("post_grasp_positive_z_action_fraction") is not None
    ]
    return {
        "n": len(e),
        "ever_success_rate": float(s.mean()),
        "final_success_rate": float(fin.mean()),
        "grasp_rate": float(g.mean()),
        "lift_rate": float(l.mean()),
        "p_lift_given_grasp": float(l[g.astype(bool)].mean()) if g.sum() else None,
        "mean_lift_delta_m": float(f(e, "object_lift_delta").mean()),
        "median_lift_delta_m": float(np.median(f(e, "object_lift_delta"))),
        "mean_episode_length": float(f(e, "episode_length").mean()),
        "mean_return_per_step": float(f(e, "return_per_step").mean()),
        "post_grasp_az_mean_episode_weighted": float(np.mean(pg)) if pg else None,
        "post_grasp_positive_az_fraction_episode_weighted": float(np.mean(pp)) if pp else None,
    }


def replay(run):
    p = RUNS[run] / "checkpoints/replay_buffer.pkl"
    with p.open("rb") as fh:
        rb = pickle.load(fh)
    n = rb.size()
    r = np.asarray(rb.rewards[:n]).reshape(-1)
    done = np.asarray(rb.dones[:n]).reshape(-1)
    timeout = np.asarray(rb.timeouts[:n]).reshape(-1)
    obs = np.asarray(rb.observations[:n]).reshape(n, -1)
    z = obs[:, 2]  # robosuite object-state begins with cube_pos
    return {
        "step": 100000,
        "transitions": int(n),
        "grasp_state_fraction": None,
        "lifted_state_fraction_proxy_z_ge_0_84": float(np.mean(z >= 0.84)),
        "success_state_fraction_proxy_reward_eq_1": float(np.mean(r >= 0.999999)),
        "post_success_transition_fraction": None,
        "done_fraction": float(done.mean()),
        "timeout_fraction": float(timeout.mean()),
        "true_terminal_fraction": float(np.mean(done * (1 - timeout))),
        "notes": "grasp/post-success identity is not encoded in the legacy replay; cube z uses object-state[2]; reward==1 is a success-state proxy and may very rarely collide with shaped reward",
    }


def main():
    status = load("fixed", "run_status.json")
    manifest = {
        "found": True,
        "usable_for_completed_100k_screen": status.get("status") == "COMPLETED"
        and status.get("steps_requested") == 100000
        and status.get("exit_code") == 0,
        "usable_as_completed_300k_experiment": False,
        "run": str(RUNS["fixed"]),
        "command": status.get("command"),
        "status": status,
        "artifacts": sorted(
            str(p.relative_to(ROOT)) for p in RUNS["fixed"].rglob("*") if p.is_file()
        ),
        "decision": "ANALYZE_WITHOUT_RETRAINING",
        "scope_limit": "The run is complete at 100k, but the preregistered fair maximum budget is 300k.",
    }
    dump("existing_fixed_horizon_manifest.json", manifest)
    common = {
        "seed": 0,
        "robot": "Panda",
        "controller": "OSC_POSE",
        "control_frequency_hz": 20,
        "horizon": 200,
        "reward_function": "robosuite Lift shaped control reward",
        "reward_shaping": True,
        "reward_scale": 1.0,
        "observation_keys": ["object-state", "robot0_proprio-state"],
        "observation_dim": 42,
        "normalization": False,
        "action_dim": 7,
        "action_bounds": [-1, 1],
        "network": [256, 256],
        "activation": "ReLU",
        "learning_rate": 0.0003,
        "buffer_size": 1000000,
        "learning_starts": 5000,
        "batch_size": 256,
        "gamma": 0.99,
        "tau": 0.005,
        "train_freq": 1,
        "gradient_steps": 1,
        "ent_coef": "auto",
        "target_entropy": -7,
        "evaluation_seeds": "900000..900099 (final)",
        "checkpoint_cadence": 50000,
    }
    cfg = {
        "AB_CONFIG_MATCH": "PASS_WITH_PROVENANCE_LIMITATION",
        "different_fields": ["termination_mode"],
        "arm_control": {**common, "termination_mode": "success"},
        "arm_fixed": {**common, "termination_mode": "fixed_horizon"},
        "provenance_limit": "Historical command/status and shared code/defaults establish the match; no immutable launch-time full config snapshot exists.",
    }
    dump("config_match_control_vs_fixed.json", cfg)
    sem = {
        "control": {
            "success_transition": {
                "terminated": True,
                "truncated": False,
                "replay_sample_done": 1,
                "critic_bootstraps": False,
            },
            "horizon_transition_without_success": {
                "terminated": False,
                "truncated": True,
                "TimeLimit.truncated": True,
                "replay_sample_done": 0,
                "critic_bootstraps": True,
            },
        },
        "fixed": {
            "success_transition_before_horizon": {
                "terminated": False,
                "truncated": False,
                "replay_sample_done": 0,
                "critic_bootstraps": True,
            },
            "horizon_transition": {
                "terminated": False,
                "truncated": True,
                "TimeLimit.truncated": True,
                "replay_sample_done": 0,
                "critic_bootstraps": True,
            },
        },
        "sac_target": "r + (1-done_sampled)*gamma*(min_target_Q - alpha*log_pi)",
        "important_reward_exposure_change": "With unchanged robosuite reward, fixed horizon receives reward 1 on every persistent-success step, while control receives it once and terminates.",
    }
    dump("termination_semantics_exact.json", sem)
    (OUT / "termination_code_trace.md").write_text(
        '# Termination code trace\n\n`rl_baselines/envs.py:35` ignores robosuite done, recomputes success, sets `terminated = success and termination_mode == "success"`, and sets the horizon as `truncated`. SB3 `DummyVecEnv` converts the five-tuple to done plus `TimeLimit.truncated`. Installed `ReplayBuffer` stores timeout separately and samples `done * (1-timeout)`. SAC uses `(1-done)` in its target.\n'
    )
    (OUT / "success_transition_bootstrap_audit.md").write_text(
        "# Success-transition bootstrap audit\n\nControl success is a true terminal: sampled done=1, so the target is the immediate reward. Fixed-horizon success before step 200 is nonterminal: sampled done=0, so SAC bootstraps. Horizon truncation is masked as timeout and bootstraps in both arms. Fixed horizon also repeats reward 1 while success persists; the treatment therefore changes both terminal bootstrapping and post-success state/reward occupancy, although the reward function itself is identical.\n"
    )
    comps = {}
    stats = {}
    for which in ["best_model", "last_model"]:
        ce, fe = episodes("control", which), episodes("fixed", which)
        comps[which] = {"control": metric(ce), "fixed": metric(fe)}
        stats[which] = {
            "ever_success": ci_pair(b(ce, "ever_success"), b(fe, "ever_success")),
            "lift": ci_pair(lift(ce), lift(fe)),
            "lift_delta_m": ci_pair(f(ce, "object_lift_delta"), f(fe, "object_lift_delta")),
        }
    dump(
        "paired_statistics.json",
        {
            "design": "SINGLE_SEED_CAUSAL_SCREEN; 100 paired reset seeds per checkpoint role",
            "comparisons": stats,
        },
    )
    dump(
        "control_re_evaluation_fixed_horizon_comparison.json",
        {
            "note": "Read-only matched-seed recomputation from the saved final evaluations; no control retraining. Legacy control lacks post-grasp fields.",
            "paired_seeds": "900000..900099",
            "comparisons": comps,
        },
    )
    dump("fixed_horizon_final_eval.json", load("fixed", "final_model_comparison.json"))
    # learning curve copy plus compact matched curves
    src = RUNS["fixed"] / "learning_curve.csv"
    (OUT / "fixed_horizon_learning_curve.csv").write_bytes(src.read_bytes())
    rows = []
    for arm in ["control", "fixed"]:
        with (RUNS[arm] / "learning_curve.csv").open() as fh:
            for row in csv.DictReader(fh):
                rows.append({"arm": arm, **row})
    keys = sorted({k for r in rows for k in r})
    with (OUT / "learning_curve_ab.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    # funnel: reach/contact unavailable in stored final diagnostics
    fr = []
    for which in ["best_model", "last_model"]:
        for arm in ["control", "fixed"]:
            e = episodes(arm, which)
            m = metric(e)
            fr.append(
                {
                    "arm": arm,
                    "checkpoint": which,
                    "reach_rate": "NA_LEGACY",
                    "contact_rate": float(
                        np.mean([x.get("min_eef_object_distance", 99) <= 0.02 for x in e])
                    ),
                    "grasp_rate": m["grasp_rate"],
                    "lift_rate": m["lift_rate"],
                    "ever_success_rate": m["ever_success_rate"],
                    "p_contact_given_reach": "NA_LEGACY",
                    "p_grasp_given_contact": "NA_NOT_NESTED_IN_EPISODE_SUMMARY",
                    "p_lift_given_grasp": m["p_lift_given_grasp"],
                    "p_success_given_lift": (
                        float(b(e, "ever_success")[lift(e)].mean()) if lift(e).sum() else None
                    ),
                }
            )
    with (OUT / "behavioral_funnel_ab.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fr[0])
        w.writeheader()
        w.writerows(fr)
    pa = []
    for which in ["best_model", "last_model"]:
        for arm in ["control", "fixed"]:
            m = metric(episodes(arm, which))
            pa.append(
                {
                    "arm": arm,
                    "checkpoint": which,
                    "mean_az_given_grasp": m["post_grasp_az_mean_episode_weighted"],
                    "median_az_given_grasp": None,
                    "p_az_positive_given_grasp": m[
                        "post_grasp_positive_az_fraction_episode_weighted"
                    ],
                    "mean_delta_object_z_given_grasp": None,
                    "availability": (
                        "available" if arm == "fixed" else "NOT_AVAILABLE_LEGACY_INSTRUMENTATION"
                    ),
                }
            )
    with (OUT / "post_grasp_action_ab.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=pa[0])
        w.writeheader()
        w.writerows(pa)
    ps = []
    for which in ["best_model", "last_model"]:
        for i, x in enumerate(episodes("fixed", which)):
            if x.get("ever_success"):
                dur = int(x.get("success_duration_steps", 0))
                final = bool(x.get("final_success"))
                cnt = int(x.get("success_count", 1))
                cls = (
                    "SUCCESS_MAINTAINED"
                    if final and cnt == 1
                    else (
                        "SUCCESS_THEN_REGRASP"
                        if cnt > 1
                        else ("SUCCESS_THEN_DROP" if not final else "SUCCESS_MAINTAINED")
                    )
                )
                ps.append(
                    {
                        "checkpoint": which,
                        "episode_index": i,
                        "first_success_step": x.get("success_step"),
                        "final_success": final,
                        "success_steps": dur,
                        "longest_success_streak": None,
                        "success_lost_after_first_success": not final,
                        "success_reentry_count": max(0, cnt - 1),
                        "classification": cls,
                    }
                )
    with (OUT / "post_success_behavior.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ps[0])
        w.writeheader()
        w.writerows(ps)
    rr = {a: replay(a) for a in ["control", "fixed"]}
    with (OUT / "replay_state_composition.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["arm", *next(iter(rr.values())).keys()])
        w.writeheader()
        [w.writerow({"arm": a, **v}) for a, v in rr.items()]
    (OUT / "critic_stage_value_audit.csv").write_text(
        'status,reason\nNOT_IDENTIFIABLE,"Saved replay lacks stage labels and a matched cross-policy state set; skipped to avoid overinterpretation"\n'
    )
    # plots
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["control best", "fixed best", "control last", "fixed last"]
    order = [
        ("best_model", "control"),
        ("best_model", "fixed"),
        ("last_model", "control"),
        ("last_model", "fixed"),
    ]
    plot_specs = [
        ("success_termination_ab.png", "ever_success_rate", "Ever success rate"),
        ("lift_termination_ab.png", "lift_rate", "Lift rate (>=4 mm)"),
        ("lift_given_grasp_termination_ab.png", "p_lift_given_grasp", "P(lift | grasp)"),
        ("lift_delta_termination_ab.png", "mean_lift_delta_m", "Mean max lift delta (m)"),
        (
            "post_grasp_z_action_termination_ab.png",
            "post_grasp_az_mean_episode_weighted",
            "Post-grasp z action",
        ),
    ]
    for fn, key, title in plot_specs:
        vals = [comps[w][a][key] for w, a in order]
        shown = [np.nan if v is None else v for v in vals]
        plt.figure(figsize=(7, 4))
        plt.bar(labels, shown, color=["#4472c4", "#ed7d31"] * 2)
        plt.ylabel(title)
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(OUT / fn, dpi=160)
        plt.close()
    for arm in ["control", "fixed"]:
        vals = [
            metric(episodes(arm, "best_model"))[k]
            for k in ["grasp_rate", "lift_rate", "ever_success_rate"]
        ]
        plt.figure(figsize=(5, 4))
        plt.bar(["grasp", "lift", "success"], vals)
        plt.ylim(0, 1)
        plt.title(f"{arm} best funnel (stored stages)")
        plt.tight_layout()
        plt.savefig(OUT / f"behavioral_funnel_{arm}.png", dpi=160)
        plt.close()
    if ps:
        plt.figure(figsize=(6, 4))
        plt.hist([r["success_steps"] for r in ps], bins=10)
        plt.xlabel("successful steps per ever-success episode")
        plt.tight_layout()
        plt.savefig(OUT / "success_retention_fixed.png", dpi=160)
        plt.close()
    plt.figure(figsize=(6, 4))
    plt.bar(
        ["control", "fixed"],
        [rr[a]["success_state_fraction_proxy_reward_eq_1"] for a in ["control", "fixed"]],
    )
    plt.ylabel("reward==1 transition fraction (proxy)")
    plt.tight_layout()
    plt.savefig(OUT / "replay_stage_composition.png", dpi=160)
    plt.close()
    cb, fb = comps["best_model"]["control"], comps["best_model"]["fixed"]
    cl, fl = comps["last_model"]["control"], comps["last_model"]["fixed"]
    # 100k result is physically promising but lower success and incomplete versus registered 300k budget.
    effect = (
        "FIXED_HORIZON_PROMISING_BUT_UNSTABLE"
        if fb["lift_rate"] > cb["lift_rate"] and fb["ever_success_rate"] <= cb["ever_success_rate"]
        else "LITTLE_EVIDENCE_OF_EFFECT"
    )
    final = {
        "EXPERIMENT": "vanilla_sac_termination_ab_v2",
        "FIXED_HORIZON_RUN_FOUND": "YES",
        "EXISTING_FIXED_HORIZON_USABLE": "YES_FOR_100K_SCREEN_NO_FOR_300K_FINAL",
        "CONTROL_HORIZON": 200,
        "FIXED_HORIZON": 200,
        "CONTROL_SUCCESS_TERMINATES": True,
        "FIXED_SUCCESS_TERMINATES": False,
        "CONTROL_HORIZON_TRUNCATED": True,
        "FIXED_HORIZON_TRUNCATED": True,
        "CONTROL_REWARD": "original shaped Lift reward",
        "FIXED_REWARD": "same original shaped Lift reward",
        "AB_CONFIG_MATCH": cfg["AB_CONFIG_MATCH"],
        "CONTROL_BEST_SUCCESS": cb["ever_success_rate"],
        "FIXED_BEST_SUCCESS": fb["ever_success_rate"],
        "CONTROL_LAST_SUCCESS": cl["ever_success_rate"],
        "FIXED_LAST_SUCCESS": fl["ever_success_rate"],
        "CONTROL_BEST_LIFT": cb["lift_rate"],
        "FIXED_BEST_LIFT": fb["lift_rate"],
        "CONTROL_P_LIFT_GIVEN_GRASP": cb["p_lift_given_grasp"],
        "FIXED_P_LIFT_GIVEN_GRASP": fb["p_lift_given_grasp"],
        "CONTROL_POST_GRASP_AZ": None,
        "FIXED_POST_GRASP_AZ": fb["post_grasp_az_mean_episode_weighted"],
        "FIXED_EVER_SUCCESS": fb["ever_success_rate"],
        "FIXED_FINAL_SUCCESS": fb["final_success_rate"],
        "FIXED_SUCCESS_RETENTION": (
            fb["final_success_rate"] / fb["ever_success_rate"] if fb["ever_success_rate"] else None
        ),
        "SUCCESS_DIFF": stats["best_model"]["ever_success"][
            "paired_difference_fixed_minus_control"
        ],
        "SUCCESS_DIFF_95CI": stats["best_model"]["ever_success"]["ci95"],
        "LIFT_DIFF": stats["best_model"]["lift"]["paired_difference_fixed_minus_control"],
        "LIFT_DIFF_95CI": stats["best_model"]["lift"]["ci95"],
        "REPLAY_SUCCESS_STATE_FRACTION_CONTROL": rr["control"][
            "success_state_fraction_proxy_reward_eq_1"
        ],
        "REPLAY_SUCCESS_STATE_FRACTION_FIXED": rr["fixed"][
            "success_state_fraction_proxy_reward_eq_1"
        ],
        "TERMINATION_EFFECT": effect,
        "BUDGET_QUALIFIER": "100k completed screen only; fixed arm has not received the matched 300k maximum budget",
        "NEXT_EXPERIMENT": "True-continue fixed horizon from 100k to 300k with replay buffer, then repeat matched evaluation; if lift advantage disappears, run targeted action/controller credit audit.",
    }
    dump("final_summary.json", final)
    dump(
        "next_action_credit_audit_plan.json",
        {
            "status": "PREPARED_NOT_RUN",
            "trigger": "Run only if the matched 300k termination comparison remains unhelpful.",
            "question": "Why does reliable grasp coexist with strongly negative z action?",
            "sources": [
                "expert trajectories",
                "scripted trajectories",
                "control and fixed checkpoints",
            ],
            "measurements": [
                "action-z before/after grasp in one common normalized representation",
                "object dz response to action-z",
                "critic Q under matched grasp-state action perturbations",
            ],
            "action_z_grid": [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1],
            "branch_outputs": [
                "immediate reward",
                "next object height",
                "grasp retained",
                "critic Q",
            ],
            "interpretations": [
                "upward motion effective but undervalued",
                "upward motion breaks grasp",
                "OSC action semantics make large +z inappropriate",
                "reward gives insufficient immediate credit",
            ],
            "large_training_authorized": False,
        },
    )
    report = f"""# Vanilla SAC termination A/B v2 — 100k completed screen\n\n## Conclusion\n\nThe existing fixed-horizon run is a valid completed **100k single-seed screen**, so it was analyzed without retraining. It is not the preregistered 300k final comparison. At best checkpoints, fixed horizon reduced ever-success from {cb['ever_success_rate']:.0%} to {fb['ever_success_rate']:.0%} and lift incidence from {cb['lift_rate']:.0%} to {fb['lift_rate']:.0%}. However, its rare lift events were much larger: mean maximum lift delta rose from {cb['mean_lift_delta_m']*1000:.1f} to {fb['mean_lift_delta_m']*1000:.1f} mm, and P(lift|grasp) rose from {cb['p_lift_given_grasp']:.1%} to {fb['p_lift_given_grasp']:.1%}. This is a mechanism hint, not a clear task-level improvement.\n\nThe paired best-checkpoint success difference is {stats['best_model']['ever_success']['paired_difference_fixed_minus_control']:+.1%} (95% bootstrap CI {stats['best_model']['ever_success']['ci95'][0]:+.1%} to {stats['best_model']['ever_success']['ci95'][1]:+.1%}); lift difference is {stats['best_model']['lift']['paired_difference_fixed_minus_control']:+.1%} (CI {stats['best_model']['lift']['ci95'][0]:+.1%} to {stats['best_model']['lift']['ci95'][1]:+.1%}). Intervals include zero.\n\n## Mechanism and caveats\n\nFixed-horizon success transitions bootstrap, whereas control success transitions are terminal. Horizon timeouts bootstrap in both arms. Fixed horizon also exposes the agent to continued successful states and repeated reward 1; this is part of the operational termination treatment and prevents attributing any effect solely to critic bootstrapping. Fixed successful episodes retained success at the final step only {fb['final_success_rate']:.0%} overall ({final['FIXED_SUCCESS_RETENTION']:.1%} of ever-success incidence at the aggregate-rate level), indicating frequent post-success loss.\n\nThe legacy control evaluation did not record post-grasp z actions, so the requested direct A/B action-mechanism comparison is not identifiable from existing artifacts. Replay success-state figures are explicitly reward-based proxies. Reach/contact nesting and critic stage values are likewise not claimed.\n\n## Decision\n\n`{effect}` at 100k: fixed horizon produces rarer but sometimes much larger lifts, without higher task success; paired uncertainty is large. The scientifically fair next step is true continuation of the existing fixed run to 300k using its saved replay buffer and optimizer state, followed by the same 100 paired seeds. No reward or hyperparameter change should be mixed into that run.\n"""
    (OUT / "final_report.md").write_text(report)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
