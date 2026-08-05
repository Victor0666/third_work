import numpy as np

def get_task_priority_v2(
    min_exec_time,
    min_comm_time,
    min_incremental_energy,
    slack,
    upward_rank,
    remaining_work,
    ready_wait_time,
    uncertainty
):
    """
    v2 hybrid: Combines Parent 2's bounded arctan urgency and slack-gated normalization
    with Parent 1's robust MAD scaling, uncertainty-aware latency inflation in tight regimes,
    and strict priority inversion for negative slack. Introduces novel *slack-conditional
    energy-efficiency dominance*: when slack < 0, energy term is suppressed; when slack >= 0,
    efficiency dominates via normalized energy-per-latency under uncertainty-inflated latency.
    Adds *wait amplification only for critical tasks* (urgency > 0.7 AND wait_time > median_wait),
    ensuring fairness without compromising DDL safety. All terms fused via slack-adaptive convex
    weights enforcing lexicographic order: DDL > energy-efficiency > fairness.
    """
    eps = 1e-08
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    # Robust MAD normalization function — from Parent 1, improved for edge cases
    def normalize_mad(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size == 1:
            return np.array([0.0])
        center = np.median(x)
        abs_devs = np.abs(x - center)
        mad = np.median(abs_devs) + eps
        if mad < eps:
            xmin, xmax = np.min(x), np.max(x)
            scale = xmax - xmin
            mad = eps if scale < eps else scale + eps
        return (x - center) / mad

    # Signed arctan urgency (Parent 2): bounded [0,1], monotonic, high sensitivity near zero slack
    arctan_urgency = (np.arctan(-slack / (1.0 + eps)) + np.pi / 2) / np.pi
    urgency_term = -5.0 * arctan_urgency  # Negative → smaller score = higher priority

    # Critical-energy density (CED): upward_rank * work / energy, but masked and weighted by slack
    base_ced = upward_rank * remaining_work / (min_incremental_energy + eps)
    # Normalize CED only over slack >= 0 to avoid violating-task dilution (Parent 2 insight)
    ced_mask = slack >= 0
    ced_valid = base_ced[ced_mask] if np.any(ced_mask) else np.array([0.0])
    norm_ced_full = np.zeros_like(slack)
    if ced_valid.size > 0:
        norm_ced_full[ced_mask] = normalize_mad(ced_valid)
    # Tightness boost: amplify CED importance as slack shrinks (more aggressive path compression)
    tightness_weight = 1.0 - arctan_urgency
    critical_energy_term = -2.2 * norm_ced_full * (1.0 + 0.6 * tightness_weight)

    # Uncertainty-aware latency inflation — only in tight regime (Parent 1), using percentile reference
    base_latency = min_exec_time + min_comm_time + eps
    median_slack_pos = np.median(slack[slack > 0]) if np.any(slack > 0) else 1.0
    tight_regime = (slack > 0) & (slack <= 2.0 * median_slack_pos)
    # Inflation factor capped and relative to 90th-percentile latency to avoid outlier skew
    lat_ref = np.percentile(base_latency, 90) + eps if base_latency.size > 1 else np.max(base_latency) + eps
    inflation_factor = np.clip(uncertainty / lat_ref, 0.0, 1.0)
    inflated_latency = np.where(tight_regime, base_latency * (1.0 + 0.3 * inflation_factor), base_latency)

    # Energy-latency efficiency: joules per second under inflated latency; normalized only over slack>=0
    eff_ratio = min_incremental_energy / (inflated_latency + eps)
    eff_mask = slack >= 0
    eff_valid = eff_ratio[eff_mask] if np.any(eff_mask) else np.array([0.0])
    norm_eff_full = np.zeros_like(slack)
    if eff_valid.size > 0:
        norm_eff_full[eff_mask] = normalize_mad(eff_valid)
    efficiency_term = 1.4 * norm_eff_full

    # Urgency-gated wait amplification (Parent 2) + criticality filter (novel):
    # Only boost waiting tasks that are both urgent (arctan_urgency > 0.7) AND have waited longer than median
    median_wait = np.median(ready_wait_time) + eps if ready_wait_time.size > 1 else np.max(ready_wait_time) + eps
    urgency_gate = arctan_urgency > 0.7
    wait_long_enough = ready_wait_time > median_wait
    clipped_wait = np.clip(ready_wait_time, 0.0, 10.0)  # Prevent runaway aging
    norm_wait_full = normalize_mad(clipped_wait)
    fairness_term = -0.35 * norm_wait_full * urgency_gate * wait_long_enough

    # Slack-dependent convex weights enforcing lexicographic priority: DDL >> energy-efficiency >> fairness
    w_urgency = np.where(slack < 0, 0.85, 0.45)
    w_energy = np.where(slack < 0, 0.05, 0.35)  # Suppress energy term on violation; emphasize on feasibility
    w_efficiency = np.where(slack < 0, 0.05, 0.15)  # Efficiency matters only when deadlines are feasible
    w_fairness = np.where(slack < 0, 0.05, 0.05)  # Minimal fairness weight under violation

    # Final score: smaller = higher priority
    score = (
        w_urgency * urgency_term +
        w_energy * critical_energy_term +
        w_efficiency * efficiency_term +
        w_fairness * fairness_term
    )

    # Clamp NaN/inf and ensure finite output
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    # Enforce shape (N,)
    score = score.reshape(-1)
    return score
