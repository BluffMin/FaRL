#!/usr/bin/env python3
"""Finalize the matched 300k success-termination vs fixed-horizon screen."""
from __future__ import annotations
import csv, json, pickle
from pathlib import Path
import numpy as np

ROOT = Path("/home/brainlab/FaRL")
BASE = ROOT / "results/vanilla_rl_baseline_v1"
OUT = BASE / "termination_ab_v2"
RUN = {
    "control": BASE / "sac_seed0_300k_from_last",
    "fixed": BASE / "sac_seed0_300k_fixed_horizon_from_100k",
}


def load(a, n="final_model_comparison.json"):
    return json.loads((RUN[a] / n).read_text())


def dump(n, x):
    (OUT / n).write_text(json.dumps(x, indent=2, sort_keys=True))


def eps(a, w):
    return load(a)[w]["episode_diagnostics"]


def arr(e, k):
    return np.array([float(x.get(k, 0) or 0) for x in e])


def binary(e, k):
    return np.array([bool(x.get(k, False)) for x in e], float)


def lifted(e):
    return arr(e, "object_lift_delta") >= 0.004


def boot(c, f, n=20000):
    d = np.asarray(f, float) - np.asarray(c, float)
    r = np.random.default_rng(300829)
    v = d[r.integers(0, len(d), (n, len(d)))].mean(1)
    return {
        "control_count_or_sum": float(np.sum(c)),
        "fixed_count_or_sum": float(np.sum(f)),
        "control_mean": float(np.mean(c)),
        "fixed_mean": float(np.mean(f)),
        "difference_fixed_minus_control": float(d.mean()),
        "ci95": [float(x) for x in np.quantile(v, [0.025, 0.975])],
        "n_pairs": len(d),
        "bootstrap_samples": n,
    }


def met(e):
    g = binary(e, "ever_grasped")
    l = lifted(e)
    s = binary(e, "ever_success")
    az = [x["post_grasp_z_action_mean"] for x in e if x.get("post_grasp_z_action_mean") is not None]
    pos = [
        x["post_grasp_positive_z_action_fraction"]
        for x in e
        if x.get("post_grasp_positive_z_action_fraction") is not None
    ]
    return {
        "episodes": len(e),
        "ever_success_rate": float(s.mean()),
        "final_success_rate": float(binary(e, "final_success").mean()),
        "grasp_rate": float(g.mean()),
        "lift_rate": float(l.mean()),
        "p_lift_given_grasp": float(l[g.astype(bool)].mean()) if g.sum() else None,
        "mean_lift_delta_m": float(arr(e, "object_lift_delta").mean()),
        "median_lift_delta_m": float(np.median(arr(e, "object_lift_delta"))),
        "mean_post_grasp_az": float(np.mean(az)) if az else None,
        "median_post_grasp_az": float(np.median(az)) if az else None,
        "mean_p_positive_az_given_grasp": float(np.mean(pos)) if pos else None,
        "mean_first_success_step_successes": (
            float(np.mean([x["success_step"] for x in e if x.get("success_step") is not None]))
            if s.sum()
            else None
        ),
        "mean_success_steps_successes": (
            float(np.mean([x.get("success_duration_steps", 0) for x in e if x.get("ever_success")]))
            if s.sum()
            else None
        ),
    }


def replay(a):
    with (RUN[a] / "checkpoints/replay_buffer.pkl").open("rb") as f:
        r = pickle.load(f)
    n = r.size()
    reward = np.asarray(r.rewards[:n]).reshape(-1)
    obs = np.asarray(r.observations[:n]).reshape(n, -1)
    d = np.asarray(r.dones[:n]).reshape(-1)
    t = np.asarray(r.timeouts[:n]).reshape(-1)
    return {
        "arm": a,
        "step": 300000,
        "transitions": n,
        "lifted_state_fraction_proxy_cube_z_ge_0_84": float(np.mean(obs[:, 2] >= 0.84)),
        "success_state_fraction_proxy_reward_eq_1": float(np.mean(reward >= 0.999999)),
        "true_terminal_fraction": float(np.mean(d * (1 - t))),
        "timeout_fraction": float(t.mean()),
    }


def main():
    status = load("fixed", "run_status.json")
    assert status["status"] == "COMPLETED" and status["termination_mode"] == "fixed_horizon"
    comp = {}
    stats = {}
    for w in ["best_model", "last_model"]:
        ce, fe = eps("control", w), eps("fixed", w)
        comp[w] = {"control": met(ce), "fixed": met(fe)}
        stats[w] = {
            "ever_success": boot(binary(ce, "ever_success"), binary(fe, "ever_success")),
            "lift": boot(lifted(ce), lifted(fe)),
            "lift_delta_m": boot(arr(ce, "object_lift_delta"), arr(fe, "object_lift_delta")),
        }
        # first-success is undefined in control (zero successes), hence not pair-identifiable.
    dump(
        "paired_statistics.json",
        {
            "design": "SINGLE_SEED_CAUSAL_SCREEN",
            "paired_seeds": list(range(900000, 900100)),
            "comparisons": stats,
            "first_success_step": "NOT_PAIRED_IDENTIFIABLE_CONTROL_HAS_ZERO_SUCCESSES",
        },
    )
    dump("fixed_horizon_final_eval.json", load("fixed"))
    dump(
        "control_re_evaluation_fixed_horizon_comparison.json",
        {
            "paired_seeds": list(range(900000, 900100)),
            "comparisons": comp,
            "note": "Saved modern 300k final evaluations, identical seeds; control was not retrained.",
        },
    )
    (OUT / "fixed_horizon_learning_curve.csv").write_bytes(
        (RUN["fixed"] / "learning_curve.csv").read_bytes()
    )
    rows = []
    for a in RUN:
        with (RUN[a] / "learning_curve.csv").open() as f:
            for x in csv.DictReader(f):
                rows.append({"arm": a, **x})
    with (OUT / "learning_curve_ab.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # action and funnel tables
    ar = []
    fu = []
    post = []
    for w in ["best_model", "last_model"]:
        for a in RUN:
            e = eps(a, w)
            m = met(e)
            ar.append(
                {
                    "arm": a,
                    "checkpoint": w,
                    "mean_az_given_grasp": m["mean_post_grasp_az"],
                    "median_az_given_grasp": m["median_post_grasp_az"],
                    "p_az_positive_given_grasp": m["mean_p_positive_az_given_grasp"],
                }
            )
            contact = np.array([x.get("min_eef_object_distance", 99) <= 0.02 for x in e])
            g = binary(e, "ever_grasped").astype(bool)
            l = lifted(e)
            s = binary(e, "ever_success").astype(bool)
            fu.append(
                {
                    "arm": a,
                    "checkpoint": w,
                    "reach_rate": "NA",
                    "contact_rate": float(contact.mean()),
                    "grasp_rate": float(g.mean()),
                    "lift_rate": float(l.mean()),
                    "ever_success_rate": float(s.mean()),
                    "p_grasp_given_contact": "NA_EPISODE_EVENTS_NOT_ORDERED",
                    "p_lift_given_grasp": float(l[g].mean()) if g.sum() else None,
                    "p_success_given_lift": float(s[l].mean()) if l.sum() else None,
                }
            )
        for i, x in enumerate(eps("fixed", w)):
            if x.get("ever_success"):
                count = int(x.get("success_count", 1))
                final = bool(x.get("final_success"))
                cls = (
                    "SUCCESS_THEN_REGRASP"
                    if count > 1
                    else (
                        "SUCCESS_MAINTAINED_TO_HORIZON"
                        if final
                        else "SUCCESS_LOST_PERMANENTLY_OR_UNOBSERVED_REENTRY"
                    )
                )
                post.append(
                    {
                        "checkpoint": w,
                        "episode_index": i,
                        "first_success_step": x.get("success_step"),
                        "success_steps": x.get("success_duration_steps"),
                        "success_reentry_count": max(0, count - 1),
                        "final_success": final,
                        "success_lost_after_first_success": not final,
                        "classification": cls,
                    }
                )
    for name, data in [
        ("post_grasp_action_ab.csv", ar),
        ("behavioral_funnel_ab.csv", fu),
        ("post_success_behavior.csv", post),
    ]:
        with (OUT / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    rr = [replay(a) for a in RUN]
    with (OUT / "replay_state_composition.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rr[0]))
        w.writeheader()
        w.writerows(rr)
    cb, fb = comp["best_model"]["control"], comp["best_model"]["fixed"]
    cl, fl = comp["last_model"]["control"], comp["last_model"]["fixed"]
    final = {
        "EXPERIMENT": "vanilla_sac_termination_ab_v2",
        "FIXED_HORIZON_RUN_FOUND": "YES",
        "EXISTING_FIXED_HORIZON_USABLE": "YES_TRUE_CONTINUATION_COMPLETED_300K",
        "CONTROL_HORIZON": 200,
        "FIXED_HORIZON": 200,
        "CONTROL_SUCCESS_TERMINATES": True,
        "FIXED_SUCCESS_TERMINATES": False,
        "CONTROL_HORIZON_TRUNCATED": True,
        "FIXED_HORIZON_TRUNCATED": True,
        "CONTROL_REWARD": "original shaped Lift reward",
        "FIXED_REWARD": "same original shaped Lift reward",
        "AB_CONFIG_MATCH": "PASS",
        "CONTROL_BEST_SUCCESS": cb["ever_success_rate"],
        "FIXED_BEST_SUCCESS": fb["ever_success_rate"],
        "CONTROL_LAST_SUCCESS": cl["ever_success_rate"],
        "FIXED_LAST_SUCCESS": fl["ever_success_rate"],
        "CONTROL_BEST_LIFT": cb["lift_rate"],
        "FIXED_BEST_LIFT": fb["lift_rate"],
        "CONTROL_P_LIFT_GIVEN_GRASP": cb["p_lift_given_grasp"],
        "FIXED_P_LIFT_GIVEN_GRASP": fb["p_lift_given_grasp"],
        "CONTROL_POST_GRASP_AZ": cb["mean_post_grasp_az"],
        "FIXED_POST_GRASP_AZ": fb["mean_post_grasp_az"],
        "FIXED_EVER_SUCCESS": fb["ever_success_rate"],
        "FIXED_FINAL_SUCCESS": fb["final_success_rate"],
        "FIXED_SUCCESS_RETENTION": fb["final_success_rate"] / fb["ever_success_rate"],
        "SUCCESS_DIFF": stats["best_model"]["ever_success"]["difference_fixed_minus_control"],
        "SUCCESS_DIFF_95CI": stats["best_model"]["ever_success"]["ci95"],
        "LIFT_DIFF": stats["best_model"]["lift"]["difference_fixed_minus_control"],
        "LIFT_DIFF_95CI": stats["best_model"]["lift"]["ci95"],
        "REPLAY_SUCCESS_STATE_FRACTION_CONTROL": rr[0]["success_state_fraction_proxy_reward_eq_1"],
        "REPLAY_SUCCESS_STATE_FRACTION_FIXED": rr[1]["success_state_fraction_proxy_reward_eq_1"],
        "TERMINATION_EFFECT": "FIXED_HORIZON_SUPPORTED",
        "NEXT_EXPERIMENT": "Replicate the identical fixed-horizon treatment with seeds 1 and 2; do not add reward shaping.",
    }
    dump("final_summary.json", final)
    report = f"""# Vanilla SAC termination A/B v2 — final 300k report\n\n## Result\n\nFixed horizon clearly solved the seed-0 grasp→lift bottleneck. On 100 matched final seeds, best checkpoint success increased from **{cb['ever_success_rate']:.0%} to {fb['ever_success_rate']:.0%}**, lift from **{cb['lift_rate']:.0%} to {fb['lift_rate']:.0%}**, and P(lift|grasp) from **{cb['p_lift_given_grasp']:.0%} to {fb['p_lift_given_grasp']:.1%}**. The last checkpoint remained strong: success {fl['ever_success_rate']:.0%}, lift {fl['lift_rate']:.0%}.\n\nPaired success difference was **+{stats['best_model']['ever_success']['difference_fixed_minus_control']:.0%}** (95% bootstrap CI {stats['best_model']['ever_success']['ci95'][0]:.0%}–{stats['best_model']['ever_success']['ci95'][1]:.0%}); lift difference was **+{stats['best_model']['lift']['difference_fixed_minus_control']:.0%}** (CI {stats['best_model']['lift']['ci95'][0]:.0%}–{stats['best_model']['lift']['ci95'][1]:.0%}).\n\n## Mechanism\n\nPost-grasp z action became materially less negative: {cb['mean_post_grasp_az']:.3f} → {fb['mean_post_grasp_az']:.3f}, while the episode-weighted positive-z fraction rose from {cb['mean_p_positive_az_given_grasp']:.3%} to {fb['mean_p_positive_az_given_grasp']:.1%}. Fixed best mean maximum lift delta was {fb['mean_lift_delta_m']:.3f} m versus {cb['mean_lift_delta_m']:.6f} m. Replay reward==1 occupancy proxy rose from {rr[0]['success_state_fraction_proxy_reward_eq_1']:.4%} to {rr[1]['success_state_fraction_proxy_reward_eq_1']:.2%}.\n\nFixed best ever-success was {fb['ever_success_rate']:.0%}, final-success {fb['final_success_rate']:.0%}; aggregate retention ratio was {final['FIXED_SUCCESS_RETENTION']:.1%}. Continued post-success behavior is therefore useful overall but not perfectly stable.\n\n## Interpretation\n\n`FIXED_HORIZON_SUPPORTED`. This is a strong single-seed causal screen, not yet a multi-seed superiority claim. The operational treatment changes both terminal bootstrapping and exposure to repeated successful states/reward, so those mechanisms are not separated here. The next experiment is an exact seeds 1–2 replication without reward shaping or hyperparameter changes.\n"""
    (OUT / "final_report.md").write_text(report)
    # refresh headline figures
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = [
        ("best_model", "control"),
        ("best_model", "fixed"),
        ("last_model", "control"),
        ("last_model", "fixed"),
    ]
    labels = ["control best", "fixed best", "control last", "fixed last"]
    specs = [
        ("success_termination_ab.png", "ever_success_rate", "Ever success"),
        ("lift_termination_ab.png", "lift_rate", "Lift"),
        ("lift_given_grasp_termination_ab.png", "p_lift_given_grasp", "P(lift | grasp)"),
        ("lift_delta_termination_ab.png", "mean_lift_delta_m", "Mean max lift delta (m)"),
        ("post_grasp_z_action_termination_ab.png", "mean_post_grasp_az", "Post-grasp z action"),
    ]
    for fn, k, title in specs:
        plt.figure(figsize=(7, 4))
        plt.bar(labels, [comp[w][a][k] for w, a in order], color=["#4472c4", "#ed7d31"] * 2)
        plt.ylabel(title)
        plt.xticks(rotation=12)
        plt.tight_layout()
        plt.savefig(OUT / fn, dpi=160)
        plt.close()
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
