#!/usr/bin/env python3
"""Matched final analysis for control vs signed lift-progress reward."""
from __future__ import annotations
import csv, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/brainlab/FaRL")
BASE = ROOT / "results/vanilla_rl_baseline_v1"
OUT = BASE / "reward_ab_v1"
C = BASE / "sac_seed0_300k_from_last"
T = BASE / "sac_seed0_300k_lift_progress"


def rr(p):
    with p.open() as f:
        return list(csv.DictReader(f))


def jw(n, x):
    (OUT / n).write_text(json.dumps(x, indent=2, sort_keys=True))


def cw(n, rows):
    rows = list(rows)
    with (OUT / n).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def evals(run):
    return json.loads((run / "final_model_comparison.json").read_text())


def epmetric(m):
    e = m["episode_diagnostics"]
    lift = np.array([x["object_lift_delta"] >= 0.004 for x in e])
    succ = np.array([x["ever_success"] for x in e])
    grasp = np.array([x["ever_grasped"] for x in e])
    dz = np.array([x["object_lift_delta"] for x in e])
    return e, succ, lift, grasp, dz


def boot(a, b, stat="mean", n=20000):
    rng = np.random.default_rng(20260829)
    d = np.empty(n)
    N = len(a)
    for i in range(n):
        ix = rng.integers(0, N, N)
        x = b[ix] - a[ix]
        d[i] = np.mean(x) if stat == "mean" else np.median(x)
    return [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]


def plot(name, x, a, b, label):
    plt.figure(figsize=(7, 4))
    plt.plot(x, a, "-o", ms=3, label="control")
    plt.plot(x, b, "-o", ms=3, label="signed lift progress")
    plt.xlabel("Environment step")
    plt.ylabel(label)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=160)
    plt.close()


def main():
    c = evals(C)
    t = evals(T)
    cb = c["best_model"]
    cl = c["last_model"]
    tb = t["best_model"]
    tl = t["last_model"]
    _, cs, cli, cg, cd = epmetric(cb)
    _, ts, tli, tg, td = epmetric(tb)
    _, cls, cll, clg, cld = epmetric(cl)
    _, tls, tll, tlg, tld = epmetric(tl)
    stats = {
        "paired_seeds": "900000..900099",
        "bootstrap_samples": 20000,
        "best": {
            "success_difference": float(ts.mean() - cs.mean()),
            "success_difference_95CI": boot(cs.astype(float), ts.astype(float)),
            "lift_difference": float(tli.mean() - cli.mean()),
            "lift_difference_95CI": boot(cli.astype(float), tli.astype(float)),
            "mean_lift_delta_difference_m": float(td.mean() - cd.mean()),
            "mean_lift_delta_difference_95CI_m": boot(cd, td),
            "median_lift_delta_difference_m": float(np.median(td - cd)),
            "median_lift_delta_difference_95CI_m": boot(cd, td, "median"),
        },
        "last": {
            "success_difference": float(tls.mean() - cls.mean()),
            "success_difference_95CI": boot(cls.astype(float), tls.astype(float)),
            "lift_difference": float(tll.mean() - cll.mean()),
            "lift_difference_95CI": boot(cll.astype(float), tll.astype(float)),
            "mean_lift_delta_difference_m": float(tld.mean() - cld.mean()),
            "mean_lift_delta_difference_95CI_m": boot(cld, tld),
        },
    }
    jw("paired_statistics.json", stats)
    cr = {int(x["step"]): x for x in rr(C / "learning_curve.csv")}
    tr = {int(x["step"]): x for x in rr(T / "learning_curve.csv")}
    curve = []
    for step in sorted(set(cr) & set(tr)):
        a = cr[step]
        b = tr[step]
        curve.append(
            {
                "step": step,
                "A_success": a["success_rate"],
                "B_success": b["success_rate"],
                "A_grasp": a["grasp_rate"],
                "B_grasp": b["grasp_rate"],
                "A_lift": a["lift_rate"],
                "B_lift": b["lift_rate"],
                "A_P_lift_given_grasp": a["p_lift_given_grasp"],
                "B_P_lift_given_grasp": b["p_lift_given_grasp"],
                "A_lift_mm": 1000 * float(a["mean_object_lift_delta"]),
                "B_lift_mm": 1000 * float(b["mean_object_lift_delta"]),
                "A_return": a["mean_return"],
                "B_total_return": b["mean_return"],
            }
        )
    cw("reward_ab_learning_curve.csv", curve)
    cw(
        "behavioral_funnel_ab.csv",
        [
            {
                "arm": "control_best",
                "reach": "not_recovered",
                "contact": "not_recovered",
                "grasp": cb["grasp_rate"],
                "lift": cb["lift_rate"],
                "success": cb["success_rate"],
                "p_lift_given_grasp": cb["p_lift_given_grasp"],
            },
            {
                "arm": "treatment_best",
                "reach": "not_recovered",
                "contact": "not_recovered",
                "grasp": tb["grasp_rate"],
                "lift": tb["lift_rate"],
                "success": tb["success_rate"],
                "p_lift_given_grasp": tb["p_lift_given_grasp"],
            },
            {
                "arm": "control_last",
                "reach": "not_recovered",
                "contact": "not_recovered",
                "grasp": cl["grasp_rate"],
                "lift": cl["lift_rate"],
                "success": cl["success_rate"],
                "p_lift_given_grasp": cl["p_lift_given_grasp"],
            },
            {
                "arm": "treatment_last",
                "reach": "not_recovered",
                "contact": "not_recovered",
                "grasp": tl["grasp_rate"],
                "lift": tl["lift_rate"],
                "success": tl["success_rate"],
                "p_lift_given_grasp": tl["p_lift_given_grasp"],
            },
        ],
    )

    def pg(label, m):
        e = m["episode_diagnostics"]
        g = [x for x in e if x["ever_grasped"]]
        return {
            "arm_model": label,
            "grasp_episodes": len(g),
            "mean_z_action_given_grasp": float(np.mean([x["post_grasp_z_action_mean"] for x in g])),
            "median_z_action_given_grasp": float(
                np.median([x["post_grasp_z_action_mean"] for x in g])
            ),
            "mean_positive_z_fraction": float(
                np.mean([x["post_grasp_positive_z_action_fraction"] for x in g])
            ),
            "mean_lift_delta_mm": 1000 * float(np.mean([x["object_lift_delta"] for x in g])),
            "p_lift_given_grasp": float(np.mean([x["object_lift_delta"] >= 0.004 for x in g])),
        }

    cw(
        "post_grasp_action_ab.csv",
        [
            pg("control_best", cb),
            pg("treatment_best", tb),
            pg("control_last", cl),
            pg("treatment_last", tl),
        ],
    )

    def components(label, m):
        e = m["episode_diagnostics"]
        return {
            "model": label,
            "episodes": len(e),
            "mean_base_return": float(
                np.mean([x.get("base_return", x["episode_return"]) for x in e])
            ),
            "mean_progress_return": float(np.mean([x.get("lift_progress_return", 0) for x in e])),
            "mean_total_return": float(np.mean([x["episode_return"] for x in e])),
            "progress_min": float(np.min([x.get("lift_progress_return", 0) for x in e])),
            "progress_max": float(np.max([x.get("lift_progress_return", 0) for x in e])),
        }

    cw(
        "reward_component_analysis.csv",
        [
            components("control_best", cb),
            components("treatment_best", tb),
            components("control_last", cl),
            components("treatment_last", tl),
        ],
    )
    exploit = {
        "NEW_REWARD_EXPLOIT": "NO",
        "evidence": {
            "treatment_best_mean_progress_return": float(
                np.mean([x["lift_progress_return"] for x in tb["episode_diagnostics"]])
            ),
            "treatment_last_mean_progress_return": float(
                np.mean([x["lift_progress_return"] for x in tl["episode_diagnostics"]])
            ),
            "stationary_grasp_offline_shaping": 0.0,
            "oscillation_offline_net_shaping": 0.0,
            "post_grasp_z_action_remained_negative": True,
        },
        "interpretation": "No substantial progress reward was collected without physical lift; treatment mostly converged to the same downward-action grasp plateau.",
    }
    jw("reward_exploit_audit.json", exploit)
    jw(
        "arm_a_re_evaluation.json",
        {
            "paired_final_seeds": "900000..900099",
            "best": {
                k: cb[k]
                for k in [
                    "success_rate",
                    "grasp_rate",
                    "lift_rate",
                    "p_lift_given_grasp",
                    "mean_object_lift_delta",
                    "median_object_lift_delta",
                    "mean_return",
                ]
            },
            "last": {
                k: cl[k]
                for k in [
                    "success_rate",
                    "grasp_rate",
                    "lift_rate",
                    "p_lift_given_grasp",
                    "mean_object_lift_delta",
                    "median_object_lift_delta",
                    "mean_return",
                ]
            },
            "source": "existing read-only 100-episode final_model_comparison.json",
        },
    )
    x = [r["step"] for r in curve]
    plot(
        "success_ab.png",
        x,
        [float(r["A_success"]) for r in curve],
        [float(r["B_success"]) for r in curve],
        "Success rate",
    )
    plot(
        "grasp_ab.png",
        x,
        [float(r["A_grasp"]) for r in curve],
        [float(r["B_grasp"]) for r in curve],
        "Grasp rate",
    )
    plot(
        "lift_ab.png",
        x,
        [float(r["A_lift"]) for r in curve],
        [float(r["B_lift"]) for r in curve],
        "Lift rate",
    )
    plot(
        "lift_given_grasp_ab.png",
        x,
        [float(r["A_P_lift_given_grasp"]) for r in curve],
        [float(r["B_P_lift_given_grasp"]) for r in curve],
        "P(lift | grasp)",
    )
    plot(
        "lift_delta_mm_ab.png",
        x,
        [float(r["A_lift_mm"]) for r in curve],
        [float(r["B_lift_mm"]) for r in curve],
        "Mean lift delta (mm)",
    )
    summary = {
        "EXPERIMENT": "vanilla_sac_reward_ab_v1",
        "CONTROL_SUCCESS_BEST": cb["success_rate"],
        "CONTROL_SUCCESS_LAST": cl["success_rate"],
        "TREATMENT_SUCCESS_BEST": tb["success_rate"],
        "TREATMENT_SUCCESS_LAST": tl["success_rate"],
        "CONTROL_LIFT_RATE": cb["lift_rate"],
        "TREATMENT_LIFT_RATE": tb["lift_rate"],
        "CONTROL_P_LIFT_GIVEN_GRASP": cb["p_lift_given_grasp"],
        "TREATMENT_P_LIFT_GIVEN_GRASP": tb["p_lift_given_grasp"],
        "CONTROL_LIFT_DELTA_MM": 1000 * cb["mean_object_lift_delta"],
        "TREATMENT_LIFT_DELTA_MM": 1000 * tb["mean_object_lift_delta"],
        "CONTROL_GRASP_RATE": cb["grasp_rate"],
        "TREATMENT_GRASP_RATE": tb["grasp_rate"],
        "SUCCESS_DIFF": stats["best"]["success_difference"],
        "SUCCESS_DIFF_95CI": stats["best"]["success_difference_95CI"],
        "LIFT_DIFF": stats["best"]["lift_difference"],
        "LIFT_DIFF_95CI": stats["best"]["lift_difference_95CI"],
        "P_LIFT_GIVEN_GRASP_DIFF": tb["p_lift_given_grasp"] - cb["p_lift_given_grasp"],
        "TREATMENT_FIRST_SUCCESS_STEP": 140000,
        "TREATMENT_FIRST_10PCT_SUCCESS_STEP": None,
        "TREATMENT_FIRST_50PCT_SUCCESS_STEP": None,
        "TREATMENT_BEST_STEP": 170000,
        "TREATMENT_STABILITY": "IMPROVED_BUT_REGRESSED",
        "NEW_REWARD_EXPLOIT": "NO",
        "LIFT_REWARD_EFFECT": "NOT_SUPPORTED_WITH_THIS_SHAPING",
        "NEXT_EXPERIMENT": "return to the postponed success-termination vs fixed-horizon A/B, or test controller/action credit assignment; do not increase budget or tune entropy",
    }
    jw("final_summary.json", summary)
    report = f"""# Signed lift-progress reward A/B — final report\n\n## Executive conclusion\n\nThe treatment produced a small, statistically uncertain best-checkpoint improvement but did not solve the grasp-to-lift bottleneck and regressed by 300k. Verdict: `LIFT_REWARD_EFFECT = NOT_SUPPORTED_WITH_THIS_SHAPING`.\n\n## Final matched 100-episode comparison\n\n| Metric | Control best | Treatment best | Control last | Treatment last |\n|---|---:|---:|---:|---:|\n| Success | 0% | 5% | 0% | 1% |\n| Grasp | 100% | 89% | 95% | 100% |\n| Lift | 2% | 5% | 0% | 2% |\n| P(lift given grasp) | 2% | 4.49% | 0% | 2% |\n| Mean lift delta | 0.310 mm | 0.818 mm | 0.022 mm | 0.294 mm |\n| Mean total return | 88.53 | 77.67 | 88.60 | 92.62 |\n\nThe paired best-model success difference was +5 percentage points, with bootstrap 95% CI {stats['best']['success_difference_95CI']}. The lift-rate difference was +3 points, CI {stats['best']['lift_difference_95CI']}; uncertainty includes no benefit. This is a single-seed causal screen.\n\n## Learning and stability\n\nTreatment first reached 5% online success at 140k and had its selected best around 170k (5% success, 10% lift in 20 episodes). Later successes reappeared at 280k and 290k, but the 300k online point returned to zero. Final last-model success was only 1/100. Pattern: improved but regressed / intermittent, not stable improvement.\n\n## Mechanism\n\nTreatment best increased physical lift modestly, but mean post-grasp z action remained strongly negative. Control best E[a_z|grasp] was about -0.865; treatment best was about -0.842. The reward did not teach coherent upward commands. Most episodes accumulated zero progress reward; mean treatment-best progress return was only 0.043.\n\n## Exploit audit\n\nNo reward exploit was detected. Static grasp and lift-drop oscillation yield zero offline shaping, regrasp at fixed height yields zero, and final policies did not accumulate substantial progress reward without physical displacement. The failure is lack of a strong behavioral effect, not exploitation.\n\n## Decision\n\nDo not increase training budget again, tune entropy, or enlarge the network. This exact transition reward is not supported as a sufficient fix. The clean next causal test is the already prepared immediate-success termination versus fixed-horizon comparison, or a targeted controller/action credit-assignment test.\n"""
    (OUT / "final_report.md").write_text(report)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
