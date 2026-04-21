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
RNG = np.random.default_rng(seed=43)

# ─── Simulation parameters ────────────────────────────────────────────────────
N_OPS        = 8      # realistic near-term operator count (south-pole region)
N_OPS_LOW    = 4      # less-dense case: fewer simultaneous rovers / landers
N_STEPS      = 200    # time steps per scenario (each step ≈ 1 min of ops)
N_RUNS       = 1_000  # Monte Carlo runs per (regime, p_link) combination
OP_HOUR_STEP = 60     # steps per operator-hour (1 step = 1 min)
P_LINK_VALS  = np.linspace(0.30, 1.00, 15)

# ─── Simple lunar south-pole geographic model ────────────────────────────────
# Represent the 80°S–90°S operating region as a circular polar map.
# A 10° latitude cap on the Moon is ~303 km in radius, so we model a
# circular south-pole operating area with preferred activity clustered on
# ridges and PSR-access corridors rather than uniformly over a square box.
POLAR_RADIUS_KM      = 303.0
INNER_EXCLUSION_KM   = 18.0   # small central exclusion to avoid overpacking the pole pixel
SHACKLETON_CENTER_KM = np.array([45.0, -35.0])
SHACKLETON_RADIUS_KM = 10.5   # ~21 km diameter crater
RIDGE_RING_R_MIN_KM  = 140.0
RIDGE_RING_R_MAX_KM  = 235.0
PSR_CLUSTER_SIGMA_KM = 22.0
HAZARD_RADIUS_KM     = 5.0
BUFFER_RADIUS_KM     = 15.0
N_LANDERS            = 3
N_LANDERS_LOW        = 2
N_LANDERS_HIGH       = 10

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


# ─── Proximity and delay helper functions ─────────────────────────────────────

def mean_proximity_scale(n_runs, n_ops, n_landers):
    """Average proximity multiplier for sampled rover/lander layouts."""
    rover_xy = sample_asset_positions(n_runs, n_ops, asset_kind="rover")
    lander_xy = sample_asset_positions(n_runs, n_landers, asset_kind="lander")
    dists = np.linalg.norm(rover_xy[:, :, np.newaxis, :] - lander_xy[:, np.newaxis, :, :], axis=3)
    proximity_scale = np.clip(
        1 - (dists - HAZARD_RADIUS_KM) / (BUFFER_RADIUS_KM - HAZARD_RADIUS_KM),
        0,
        1,
    )
    max_scale = proximity_scale.max(axis=2)
    return float(max_scale.mean())


def geographic_delay_curve(delay_fn, p_link_vals, *, n_ops, n_landers, traffic_scale, reroute_bonus=0.0):
    """Delay curve augmented by geographic proximity and reroute burden."""
    mean_scale = mean_proximity_scale(N_RUNS, n_ops, n_landers)
    inside_hazard_share = max(0.0, (mean_scale - 0.15) / 0.85)
    near_hazard_share = max(0.0, mean_scale - inside_hazard_share)

    base_delay = np.array([
        delay_fn(p, traffic_scale=traffic_scale)
        for p in p_link_vals
    ])

    proximity_delay = 10.0 * near_hazard_share + 18.0 * inside_hazard_share
    reroute_delay = reroute_bonus * (12.0 * near_hazard_share + 20.0 * inside_hazard_share)

    return base_delay + proximity_delay + reroute_delay

# ─── Simple geographic sampling ──────────────────────────────────────────────
def sample_asset_positions(n_runs, n_assets, asset_kind="rover"):
    """Sample asset positions on a circular lunar south-pole map.

    Rovers are biased toward a ridge/corridor ring where traverse activity is
    more likely. Landers are biased toward a Shackleton-adjacent cluster plus a
    broader ridge ring. All points are constrained to remain inside the polar
    cap and outside the crater interior.
    """
    pts = np.zeros((n_runs, n_assets, 2), dtype=float)

    for i in range(n_runs):
        placed = 0
        while placed < n_assets:
            if asset_kind == "lander":
                # Mix: 65% near the Shackleton-adjacent operating cluster,
                # 35% on the broader south-pole ridge ring.
                if RNG.random() < 0.65:
                    cand = SHACKLETON_CENTER_KM + RNG.normal(0.0, PSR_CLUSTER_SIGMA_KM, size=2)
                else:
                    theta = RNG.uniform(0.0, 2.0 * np.pi)
                    r = RNG.uniform(RIDGE_RING_R_MIN_KM, RIDGE_RING_R_MAX_KM)
                    cand = np.array([r * np.cos(theta), r * np.sin(theta)])
            else:
                # Rovers are more corridor-like: keep them concentrated on the
                # accessible ridge ring around the south pole.
                theta = RNG.uniform(0.0, 2.0 * np.pi)
                r = RNG.uniform(RIDGE_RING_R_MIN_KM, RIDGE_RING_R_MAX_KM)
                cand = np.array([r * np.cos(theta), r * np.sin(theta)])

            r_pole = np.linalg.norm(cand)
            d_shackleton = np.linalg.norm(cand - SHACKLETON_CENTER_KM)

            inside_polar_cap = (INNER_EXCLUSION_KM <= r_pole <= POLAR_RADIUS_KM)
            outside_shackleton = d_shackleton >= SHACKLETON_RADIUS_KM

            if inside_polar_cap and outside_shackleton:
                pts[i, placed] = cand
                placed += 1

    return pts

# ─── Monte Carlo sampling ─────────────────────────────────────────────────────
def run_scenario(conflict_prob_fn, p_link, n_ops, n_landers, traffic_scale=1.0, n_runs=N_RUNS):
    """
    Each run samples rover and lander positions on a circular lunar south-pole
    map. Conflict probability is then scaled by each rover's geographic proximity to
    """

    rover_xy = sample_asset_positions(n_runs, n_ops, asset_kind="rover")
    lander_xy = sample_asset_positions(n_runs, n_landers, asset_kind="lander")

    # Compute distances from each rover to each lander (shape: n_runs x n_ops x n_landers)
    dists = np.linalg.norm(rover_xy[:, :, np.newaxis, :] - lander_xy[:, np.newaxis, :, :], axis=3)

    # Conflict probability scaling: closer proximity increases conflict risk
    # Scale factor: 1 if distance <= HAZARD_RADIUS_KM, linearly decreasing to 0 at BUFFER_RADIUS_KM
    proximity_scale = np.clip(1 - (dists - HAZARD_RADIUS_KM) / (BUFFER_RADIUS_KM - HAZARD_RADIUS_KM), 0, 1)
    max_scale = proximity_scale.max(axis=2)  # max proximity scale per rover

    # Per-step probability consistent with the run-level conflict probability
    base_conflict_prob = conflict_prob_fn(p_link, traffic_scale=traffic_scale)
    per_step_p = 1 - (1 - base_conflict_prob) ** (1 / (n_ops * N_STEPS / 10))

    # Shape: (n_runs, n_ops, N_STEPS)
    conflicts = RNG.random((n_runs, n_ops, N_STEPS)) < (per_step_p * max_scale[:, :, np.newaxis])

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
    "less_dense": {
        "label": "Less-dense environment",
        "n_ops": N_OPS_LOW,
        "n_landers": N_LANDERS_LOW,
        "traffic_scale": 0.55,
        "outfile": "plot1_conflict_rate_less_dense.png",
    },
    "nominal": {
        "label": "Nominal density",
        "n_ops": N_OPS,
        "n_landers": N_LANDERS,
        "traffic_scale": 1.0,
        "outfile": "plot1_conflict_rate.png",
    },
    "high_density": {
        "label": "High-density environment",
        "n_ops": 50,
        "n_landers": N_LANDERS_HIGH,
        "traffic_scale": 1.25,
        "outfile": "plot1_conflict_rate_50_ops.png",
    },
}


def compute_results(n_ops, n_landers, traffic_scale):
    results = {}
    for label, fn in regimes.items():
        frac_list, sem_frac_list = [], []
        oph_list,  sem_oph_list  = [], []
        for p in P_LINK_VALS:
            fr, sfr, oph, soph = run_scenario(
                fn,
                p,
                n_ops=n_ops,
                n_landers=n_landers,
                traffic_scale=traffic_scale,
                n_runs=N_RUNS,
            )
            frac_list.append(fr);  sem_frac_list.append(sfr)
            oph_list.append(oph);  sem_oph_list.append(soph)
        results[label] = {
            "frac":     np.array(frac_list),
            "sem_frac": np.array(sem_frac_list),
            "oph":      np.array(oph_list),
            "sem_oph":  np.array(sem_oph_list),
            "delay": geographic_delay_curve(
                mean_delay_no_notice if "No Notice" in label else mean_delay_notlo,
                P_LINK_VALS,
                n_ops=n_ops,
                n_landers=n_landers,
                traffic_scale=traffic_scale,
                reroute_bonus=1.0 if "NOTLO" in label else 0.0,
            ),
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

    frac_max = max(float(np.max(res["frac"] + 1.96 * res["sem_frac"])) for res in results.values())
    frac_ylim_top = max(0.05, min(1.05, frac_max * 1.35))
    ax_frac.axvline(x=0.75, color="grey", linestyle="--", linewidth=1.2, zorder=1)
    ax_frac.text(0.76, frac_ylim_top * 0.97, "FAA NAS\noutage\nthreshold\n(~75 % link)",
                 fontsize=7.5, color="grey", va="top")


    ax_frac.set_xlabel("Communication Link Reliability  $p_{link}$", fontsize=10)
    ax_frac.set_ylabel("Fraction of runs with ≥1\ndust/plume hazard exposure", fontsize=9.5)
    ax_frac.set_title("(A)  Hazard Exposure Rate", fontsize=10.5, fontweight="bold", pad=8)
    ax_frac.set_xlim(0.28, 1.02)
    ax_frac.set_ylim(0.0, frac_ylim_top)
    ax_frac.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax_frac.legend(fontsize=8.5, loc="upper right", framealpha=0.9)
    ax_frac.grid(True, linestyle=":", alpha=0.5)


    # ─── Panel B: expected conflicts per operator-hour ───────────────────────
    for label, res in results.items():
        s = STYLE[label]
        ax_oph.plot(P_LINK_VALS, res["oph"],
                    color=s["color"], marker=s["marker"], markersize=7,
                    linewidth=2.2, label=label, zorder=3)

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
        f"($N_{{ops}}$={n_ops}, circular south-pole geography, 1 000 Monte Carlo runs)",
        fontsize=12, fontweight="bold",
    )

    plt.savefig(outfile, dpi=180, bbox_inches="tight")
    print(f"Saved: {outfile}")
    plt.show()


# === Combined hazard and delay figure functions ===

def make_combined_hazard_figure(all_case_results, outfile="plot1_hazard_exposure_all_cases.png"):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=False)
    fig.subplots_adjust(top=0.82, bottom=0.16, wspace=0.32)

    case_order = ["less_dense", "nominal", "high_density"]
    global_frac_ylim_top = max(
        0.05,
        min(
            1.05,
            max(float(np.max(res["frac"])) for case_results in all_case_results.values() for res in case_results.values()) * 1.35,
        ),
    )

    for ax, case_key in zip(axes, case_order):
        case = DENSITY_CASES[case_key]
        results = all_case_results[case_key]

        for label, res in results.items():
            s = STYLE[label]
            ax.plot(
                P_LINK_VALS,
                res["frac"],
                color=s["color"],
                marker=s["marker"],
                markersize=6,
                linewidth=2.0,
                label=label,
            )

        # frac_max = max(float(np.max(res["frac"])) for res in results.values())
        # frac_ylim_top = max(0.05, min(1.05, frac_max * 1.35))
        frac_ylim_top = global_frac_ylim_top
        ax.axvline(x=0.75, color="grey", linestyle="--", linewidth=1.1)
        ax.text(
            0.76,
            frac_ylim_top * 0.97,
            "FAA NAS\noutage\nthreshold\n(~75 % link)",
            fontsize=7.0,
            color="grey",
            va="top",
        )
        ax.set_title(
            f"{case['label']}\n$N_{{ops}}$={case['n_ops']}",
            fontsize=10.5,
            fontweight="bold",
            pad=8,
        )
        ax.set_xlim(0.28, 1.02)
        ax.set_ylim(0.0, frac_ylim_top)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.set_xlabel("Communication Link Reliability  $p_{link}$", fontsize=9.5)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    for ax in axes:
        ax.set_ylabel("Fraction of runs with ≥1\ndust/plume hazard exposure", fontsize=9.5)
    axes[-1].legend(fontsize=8.5, loc="upper right", framealpha=0.9)

    fig.suptitle(
        "Plot 1A – Hazard Exposure Rate Across Operational Density Cases",
        fontsize=12,
        fontweight="bold",
    )
    plt.savefig(outfile, dpi=180, bbox_inches="tight")
    print(f"Saved: {outfile}")
    plt.show()



def make_combined_delay_figure(all_case_results, outfile="plot1_safety_burden_all_cases.png"):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=False)
    fig.subplots_adjust(top=0.82, bottom=0.16, wspace=0.32)

    case_order = ["less_dense", "nominal", "high_density"]
    global_delay_ylim_top = max(
        float(np.max(res["delay"])) for case_results in all_case_results.values() for res in case_results.values()
    ) * 1.12

    for ax, case_key in zip(axes, case_order):
        case = DENSITY_CASES[case_key]
        results = all_case_results[case_key]

        for label, res in results.items():
            s = STYLE[label]
            ax.plot(
                P_LINK_VALS,
                res["delay"],
                color=s["color"],
                marker=s["marker"],
                markersize=5,
                linewidth=2.0,
                label=label,
            )

        delay_no = results["No Notice (Baseline)"]["delay"]
        delay_nlo = results["NOTLO (Proposed)"]["delay"]
        ax.fill_between(P_LINK_VALS, delay_nlo, delay_no,
                        alpha=0.10, color="#1F77B4",
                        label="_nolegend_")
        ax.axvline(x=0.75, color="grey", linestyle="--", linewidth=1.0)
        ax.text(
            0.62,
            (delay_no[7] + delay_nlo[7]) / 2,
            "NOTLO\nsaves delay\nvs. reactive\nstops",
            fontsize=7,
            color="#1F77B4",
            ha="center",
            va="center",
        )
        ax.set_title(
            f"{case['label']}\n$N_{{ops}}$={case['n_ops']}",
            fontsize=10.5,
            fontweight="bold",
            pad=8,
        )
        ax.set_xlim(0.28, 1.02)
        ax.set_ylim(0.0, global_delay_ylim_top)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.set_xlabel("Communication Link Reliability  $p_{link}$", fontsize=9.5)
    for ax in axes:
        ax.set_ylabel("Mean operator delay (min)", fontsize=9.5)
    axes[-1].legend(fontsize=8.5, loc="upper right", framealpha=0.9)

    fig.suptitle(
        "Plot 1B – Safety vs. Burden Across Operational Density Cases",
        fontsize=12,
        fontweight="bold",
    )
    plt.savefig(outfile, dpi=180, bbox_inches="tight")
    print(f"Saved: {outfile}")
    plt.show()



all_case_results = {}
for case_key, case in DENSITY_CASES.items():
    results = compute_results(case["n_ops"], case["n_landers"], case["traffic_scale"])
    all_case_results[case_key] = results
    make_figure(
        results,
        density_label=case["label"],
        n_ops=case["n_ops"],
        outfile=case["outfile"],
    )

make_combined_hazard_figure(all_case_results)
make_combined_delay_figure(all_case_results)
