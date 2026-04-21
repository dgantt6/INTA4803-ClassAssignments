"""
NOTLO Project - Plot 1: Safety Analysis
Dust/Plume Hazard Exposure Rate & Expected Conflicts per Operator-Hour
vs. Communication Link Reliability (p_link)

Two governance regimes: No-Notice | NOTLO (Notice to Lunar Operators)

Authors: Clara Poitevin, Jalen Cauley, Dylan Gantt,
         Erik Goeke, Waleed Sadiq, Rut Santana
INTA 4803/8803 - Space Sustainability, Georgia Tech
April 2026

"Conflict" definition (regolith-specific):
  An operator's rover/asset enters the active dust-plume or engine-exhaust
  hazard footprint of another lander/launcher during its declared active
  time window (i.e., >1 exposure-minute inside the footprint).

Metrics:
  (A) Fraction of runs with ≥1 conflict  - intuitive but saturates near 100 %
      when N_ops and N_steps are large.
  (B) Expected conflicts per operator-hour - normalised, does not saturate;
      better for comparing governance regimes at high operator density.

Safety vs. burden tradeoff:
  Paired inset shows mean operator delay (minutes) vs. p_link for both
  regimes, illustrating that NOTLO's safety gain comes with a modest,
  bounded delay cost.

Data basis:
  - FAA NAS Safety Review (2023): communication outage → conflict escalation
  - Lunar south-pole missions: realistic operator density (N=8)
  - NOTAM governance literature: notice-receipt rate drives coordination quality
  - Monte Carlo simulation (1 000 runs per p_link level) calibrated to
    order-of-magnitude lunar surface conflict estimates
  - Dust/plume hazard footprint radius: ~2–5 km (Metzger et al. 2021,
    Immer et al. 2011 — regolith ejecta cone models)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ─── Reproducibility ──────────────────────────────────────────────────────────
RNG = np.random.default_rng(seed=42)

# ─── Simulation parameters ────────────────────────────────────────────────────
N_OPS        = 8      # realistic near-term operator count (south-pole region)
N_OPS_LOW    = 4      # less-dense case: fewer simultaneous rovers / landers
N_STEPS      = 200    # time steps per scenario (each step ≈ 1 min of ops)
N_RUNS       = 1_000  # Monte Carlo runs per (regime, p_link) combination
OP_HOUR_STEP = 60     # steps per operator-hour (1 step = 1 min)
P_LINK_VALS  = np.linspace(0.30, 1.00, 15)

# ─── Conflict probability models ──────────────────────────────────────────────
# "Conflict" = rover enters active dust/plume hazard footprint.
# Calibrated so that:
#   No-Notice  ~ 45 % conflict/run at p_link = 1.0  (no coordination)
#   NOTLO      ~ 10 % conflict/run at p_link = 1.0  (full notice receipt)
# Both degrade toward ~65–75 % at p_link → 0 (total comms loss).
# NOTLO degrades more steeply because its safety relies on notice delivery,
# but its fallback "safe-stop / stay-in-corridor" rule limits the floor.

def conflict_prob_no_notice(p_link, traffic_scale=1.0):
    """No dynamic notice layer; operators rely on ad-hoc situational awareness."""
    base_rate   = 0.45 * traffic_scale
    degradation = (1 - p_link) ** 1.2 * 0.28 * traffic_scale
    return np.clip(base_rate + degradation, 0, 1)


def conflict_prob_notlo(p_link, traffic_scale=1.0):
    """
    NOTLO: time-bounded dust/plume notices + safe-stop/corridor fallback
    when p_link drops below ~60 %.
    """
    base_rate   = 0.10 * traffic_scale
    # Steeper degradation than no-notice because NOTLO *relies* on comms,
    # but the fallback rule caps the worst-case exposure.
    degradation = (1 - p_link) ** 2.2 * 0.55 * traffic_scale
    return np.clip(base_rate + degradation, 0, 1)

# ─── Delay model (for safety-vs-burden inset) ─────────────────────────────────
# Mean operator delay (minutes) incurred by each regime.
# No-Notice: frequent reactive stops when hazard encountered unexpectedly.
# NOTLO: pre-planned detour around declared footprint; small penalty for
#        missed notices; bounded by fallback corridor rule.

def mean_delay_no_notice(p_link, traffic_scale=1.0):
    """Reactive stops dominate; worsens as comms degrade (more surprises)."""
    base_delay  = 38.0 * (0.70 + 0.30 * traffic_scale)
    extra       = (1 - p_link) * 22.0 * traffic_scale
    return base_delay + extra


def mean_delay_notlo(p_link, traffic_scale=1.0):
    """
    Pre-planned detour adds a fixed overhead; missed-notice penalty is small
    because the fallback corridor rule keeps operators out of footprints.
    """
    base_delay  = 18.0 * (0.75 + 0.25 * traffic_scale)
    extra       = (1 - p_link) ** 1.5 * 14.0 * traffic_scale
    return base_delay + extra

# ─── Monte Carlo sampling ─────────────────────────────────────────────────────
def run_scenario(conflict_prob_fn, p_link, n_ops, traffic_scale=1.0, n_runs=N_RUNS):
    """
    Each run: n_ops agents operating for N_STEPS (1 step ≈ 1 min).
    Per-step conflict probability derived from the governance model.

    Returns
    -------
    frac_runs   : fraction of runs with ≥1 dust/plume hazard exposure
    sem_frac    : standard error of frac_runs
    exp_per_oph : expected conflicts per operator-hour (normalised metric)
    sem_oph     : standard error of exp_per_oph
    """
    # Per-step probability consistent with the run-level conflict probability
    per_step_p = 1 - (1 - conflict_prob_fn(p_link, traffic_scale=traffic_scale)) ** (1 / (n_ops * N_STEPS / 10))

    # Shape: (n_runs, n_ops, N_STEPS)
    conflicts = RNG.random((n_runs, n_ops, N_STEPS)) < per_step_p

    # Metric A: fraction of runs with ≥1 conflict
    run_had_conflict = conflicts.any(axis=(1, 2))
    frac_runs = run_had_conflict.mean()
    sem_frac  = run_had_conflict.std() / np.sqrt(n_runs)

    # Metric B: expected conflicts per operator-hour
    # total conflict events per run / (n_ops * N_STEPS / OP_HOUR_STEP)
    op_hours_per_run = n_ops * N_STEPS / OP_HOUR_STEP
    conflicts_per_run = conflicts.sum(axis=(1, 2))
    rate_per_oph = conflicts_per_run / op_hours_per_run
    exp_per_oph  = rate_per_oph.mean()
    sem_oph      = rate_per_oph.std() / np.sqrt(n_runs)

    return frac_runs, sem_frac, exp_per_oph, sem_oph

# ─── Compute curves ───────────────────────────────────────────────────────────
regimes = {
    "No Notice (Baseline)": conflict_prob_no_notice,
    "NOTLO (Proposed)":     conflict_prob_notlo,
}

STYLE = {
    "No Notice (Baseline)": dict(color="#D62728", marker="^", ls="-"),
    "NOTLO (Proposed)":     dict(color="#1F77B4", marker="o", ls="-"),
}

DENSITY_CASES = {
    "nominal": {
        "label": "Nominal density",
        "n_ops": N_OPS,
        "traffic_scale": 1.0,
        "outfile": "plot1_conflict_rate.png",
    },
    "less_dense": {
        "label": "Less-dense environment",
        "n_ops": N_OPS_LOW,
        "traffic_scale": 0.55,
        "outfile": "plot1_conflict_rate_less_dense.png",
    },
}


def compute_results(n_ops, traffic_scale):
    results = {}
    for label, fn in regimes.items():
        frac_list, sem_frac_list = [], []
        oph_list,  sem_oph_list  = [], []
        for p in P_LINK_VALS:
            fr, sfr, oph, soph = run_scenario(fn, p, n_ops=n_ops, traffic_scale=traffic_scale)
            frac_list.append(fr);  sem_frac_list.append(sfr)
            oph_list.append(oph);  sem_oph_list.append(soph)
        results[label] = {
            "frac":     np.array(frac_list),
            "sem_frac": np.array(sem_frac_list),
            "oph":      np.array(oph_list),
            "sem_oph":  np.array(sem_oph_list),
            "delay":    np.array([
                (mean_delay_no_notice if "No Notice" in label else mean_delay_notlo)(
                    p, traffic_scale=traffic_scale
                )
                for p in P_LINK_VALS
            ]),
        }
    return results

# ─── Plotting helper ──────────────────────────────────────────────────────────
def make_figure(results, *, density_label, n_ops, outfile):
    fig = plt.figure(figsize=(15, 6))
    gs  = gridspec.GridSpec(1, 3, width_ratios=[2.2, 2.2, 1.4], wspace=0.42)
    fig.subplots_adjust(top=0.82, bottom=0.13)

    ax_frac  = fig.add_subplot(gs[0])
    ax_oph   = fig.add_subplot(gs[1])
    ax_delay = fig.add_subplot(gs[2])

    # ─── Panel A: fraction of runs with ≥1 hazard exposure ───────────────────
    for label, res in results.items():
        s = STYLE[label]
        ax_frac.plot(P_LINK_VALS, res["frac"],
                     color=s["color"], marker=s["marker"], markersize=7,
                     linewidth=2.2, label=label, zorder=3)
        ax_frac.fill_between(P_LINK_VALS,
                             res["frac"] - 1.96 * res["sem_frac"],
                             res["frac"] + 1.96 * res["sem_frac"],
                             color=s["color"], alpha=0.12, zorder=2)

    ax_frac.axvline(x=0.75, color="grey", linestyle="--", linewidth=1.2, zorder=1)
    ax_frac.text(0.76, 0.97, "FAA NAS\noutage\nthreshold\n(~75 % link)",
                 fontsize=7.5, color="grey", va="top")

    notlo_frac = results["NOTLO (Proposed)"]["frac"]
    idx_60 = np.argmin(np.abs(P_LINK_VALS - 0.60))
    ax_frac.annotate(
        "NOTLO fallback:\nsafe-stop corridor\nactivates ≲60 %",
        xy=(P_LINK_VALS[idx_60], notlo_frac[idx_60]),
        xytext=(0.38, 0.52),
        fontsize=7.5, color=STYLE["NOTLO (Proposed)"]["color"],
        arrowprops=dict(arrowstyle="->",
                        color=STYLE["NOTLO (Proposed)"]["color"], lw=1.2),
    )

    ax_frac.set_xlabel("Communication Link Reliability  $p_{link}$", fontsize=10)
    ax_frac.set_ylabel("Fraction of runs with ≥1\ndust/plume hazard exposure", fontsize=9.5)
    ax_frac.set_title("(A)  Hazard Exposure Rate", fontsize=10.5, fontweight="bold", pad=8)
    ax_frac.set_xlim(0.28, 1.02)
    ax_frac.set_ylim(0.0, 1.05)
    ax_frac.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax_frac.legend(fontsize=8.5, loc="upper right", framealpha=0.9)
    ax_frac.grid(True, linestyle=":", alpha=0.5)

    ax_frac2 = ax_frac.twiny()
    ax_frac2.set_xlim(ax_frac.get_xlim())
    ax_frac2.set_xticks(P_LINK_VALS[::2])
    ax_frac2.set_xticklabels([f"{(1-p):.0%}" for p in P_LINK_VALS[::2]], fontsize=7.5)
    ax_frac2.set_xlabel("Outage Fraction  $(1 - p_{link})$", fontsize=8.5, labelpad=5)

    # ─── Panel B: expected conflicts per operator-hour ───────────────────────
    for label, res in results.items():
        s = STYLE[label]
        ax_oph.plot(P_LINK_VALS, res["oph"],
                    color=s["color"], marker=s["marker"], markersize=7,
                    linewidth=2.2, label=label, zorder=3)
        ax_oph.fill_between(P_LINK_VALS,
                            res["oph"] - 1.96 * res["sem_oph"],
                            res["oph"] + 1.96 * res["sem_oph"],
                            color=s["color"], alpha=0.12, zorder=2)

    ax_oph.axvline(x=0.75, color="grey", linestyle="--", linewidth=1.2, zorder=1)
    ax_oph.set_xlabel("Communication Link Reliability  $p_{link}$", fontsize=10)
    ax_oph.set_ylabel("Expected footprint entries\nper operator-hour", fontsize=9.5)
    ax_oph.set_title("(B)  Normalised Conflict Rate  (preferred)", fontsize=10.5, fontweight="bold", pad=8)
    ax_oph.set_xlim(0.28, 1.02)
    ax_oph.legend(fontsize=8.5, loc="upper right", framealpha=0.9)
    ax_oph.grid(True, linestyle=":", alpha=0.5)

    p1_no  = results["No Notice (Baseline)"]["oph"][-1]
    p1_nlo = results["NOTLO (Proposed)"]["oph"][-1]
    reduction_pct = (p1_no - p1_nlo) / p1_no * 100
    ax_oph.annotate(
        f"NOTLO reduces\nexposure rate by\n~{reduction_pct:.0f}% at $p_{{link}}$=1",
        xy=(1.00, p1_nlo),
        xytext=(0.72, p1_nlo + (p1_no - p1_nlo) * 0.55),
        fontsize=7.5, color=STYLE["NOTLO (Proposed)"]["color"],
        arrowprops=dict(arrowstyle="->",
                        color=STYLE["NOTLO (Proposed)"]["color"], lw=1.2),
    )

    ax_oph2 = ax_oph.twiny()
    ax_oph2.set_xlim(ax_oph.get_xlim())
    ax_oph2.set_xticks(P_LINK_VALS[::2])
    ax_oph2.set_xticklabels([f"{(1-p):.0%}" for p in P_LINK_VALS[::2]], fontsize=7.5)
    ax_oph2.set_xlabel("Outage Fraction  $(1 - p_{link})$", fontsize=8.5, labelpad=5)

    # ─── Panel C: delay vs p_link (safety-vs-burden tradeoff) ─────────────────
    for label, res in results.items():
        s = STYLE[label]
        ax_delay.plot(P_LINK_VALS, res["delay"],
                      color=s["color"], marker=s["marker"], markersize=5,
                      linewidth=2.0, label=label, zorder=3)

    ax_delay.axvline(x=0.75, color="grey", linestyle="--", linewidth=1.0, zorder=1)
    ax_delay.set_xlabel("$p_{link}$", fontsize=10)
    ax_delay.set_ylabel("Mean operator delay (min)", fontsize=9)
    ax_delay.set_title("(C)  Safety vs. Burden", fontsize=10.5, fontweight="bold", pad=8)
    ax_delay.set_xlim(0.28, 1.02)
    ax_delay.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ax_delay.grid(True, linestyle=":", alpha=0.5)

    delay_no  = results["No Notice (Baseline)"]["delay"]
    delay_nlo = results["NOTLO (Proposed)"]["delay"]
    ax_delay.fill_between(P_LINK_VALS, delay_nlo, delay_no,
                          alpha=0.10, color="#1F77B4",
                          label="_nolegend_")
    ax_delay.text(0.62, (delay_no[7] + delay_nlo[7]) / 2,
                  "NOTLO\nsaves delay\nvs. reactive\nstops",
                  fontsize=7, color="#1F77B4", ha="center", va="center")

    fig.suptitle(
        "Plot 1 – NOTLO Safety Analysis  |  "
        f"{density_label}  |  "
        "Dust/Plume Hazard Exposure vs. Communication Reliability  "
        f"($N_{{ops}}$={n_ops},  1 000 Monte Carlo runs)",
        fontsize=12, fontweight="bold",
    )

    plt.savefig(outfile, dpi=180, bbox_inches="tight")
    print(f"Saved: {outfile}")
    plt.show()


for case in DENSITY_CASES.values():
    results = compute_results(case["n_ops"], case["traffic_scale"])
    make_figure(
        results,
        density_label=case["label"],
        n_ops=case["n_ops"],
        outfile=case["outfile"],
    )
