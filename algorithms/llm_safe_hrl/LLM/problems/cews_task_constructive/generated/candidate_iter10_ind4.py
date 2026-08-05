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
    v2 mutation: Replaces adaptive IQR normalization with robust MAD-based scaling;
    introduces *slack-gated urgency ratio* (1/(|slack|+eps) for slack < 0, arctan(-slack) for slack >= 0);
    replaces critical-energy density with *energy-efficiency-adjusted path criticality*:
        (upward_rank * remaining_work) / (min_incremental_energy + eps) * (1 + sigmoid(slack/tau));
    uses *uncertainty-aware latency inflation* only in tight-schedule regime (0 < slack <= 2*median_slack);
    replaces sqrt-wait fairness with *linearly clipped wait boost*, scaled by slack abundance threshold;
    enforces strict priority inversion: negative slack → dominant urgency term; zero energy contribution when infeasible;
    all terms fused via convex combination with slack-dependent weights to enforce lexicographic DDL > energy > fairness.
    """
    eps = 1e-8
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    # Robust normalization: MAD-based, degenerate-safe
    def normalize_mad(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size == 1:
            return np.array([0.0])
        center = np.median(x)
        abs_devs = np.abs(x - center)
        mad = np.median(abs_devs) + eps
        # Avoid flat arrays: fallback to range if MAD ≈ 0
        if mad < eps:
            xmin, xmax = np.min(x), np.max(x)
            scale = xmax - xmin
            mad = eps if scale < eps else scale + eps
        return (x - center) / mad

    # Urgency: piecewise smooth — dominant penalty for negative slack, bounded monotonic for feasible
    urgency_raw = np.where(
        slack < 0,
        1.0 / (np.abs(slack) + eps),  # strong inverse penalty for violation risk
        np.arctan(-slack)  # bounded [-π/2, 0] for slack >= 0
    )
    norm_urgency = normalize_mad(urgency_raw)
    # Scale urgency to dominate when slack < 0: higher weight, no cancellation
    urgency_term = -3.5 * norm_urgency

    # Energy-efficiency-adjusted path criticality: boosted by slack feasibility margin
    tau_slack = np.percentile(np.abs(slack[slack != 0]), 50) if np.any(slack != 0) else 1.0
    tau_slack = max(tau_slack, eps)
    feasibility_boost = 1.0 + 1.0 / (1.0 + np.exp(-slack / tau_slack))  # sigmoid ∈ [1, 2]
    ced_base = upward_rank * remaining_work / (min_incremental_energy + eps)
    ced_masked = np.where(slack >= 0, ced_base * feasibility_boost, 0.0)
    norm_ced = normalize_mad(ced_masked)
    ced_term = -1.0 * norm_ced

    # Latency penalty: only active in *tight-but-feasible* regime (0 < slack <= 2*median_slack)
    median_slack = np.median(slack[slack > 0]) if np.any(slack > 0) else 1.0
    tight_mask = (slack > 0) & (slack <= 2.0 * median_slack)
    base_latency = min_exec_time + min_comm_time + eps
    uncertainty_inflation = np.where(
        tight_mask,
        base_latency * (1.0 + uncertainty / (np.mean(uncertainty) + eps)),
        base_latency
    )
    norm_latency = normalize_mad(uncertainty_inflation)
    latency_term = 0.7 * norm_latency

    # Fairness: linear wait boost, clipped and scaled by slack abundance
    wait_boost = np.clip(ready_wait_time, 0.0, 10.0)  # cap extreme aging
    # Stronger fairness only when slack is abundant: scale inversely with urgency magnitude
    slack_abundance = np.where(slack > median_slack, 1.0 + (slack - median_slack) / (median_slack + eps), 1.0)
    fairness_raw = wait_boost / (slack_abundance + eps)
    norm_fairness = normalize_mad(fairness_raw)
    fairness_term = -0.15 * norm_fairness

    # Lexicographic fusion: urgency dominates when slack < 0; otherwise balance CED & latency
    # Weight vector: [urgency_weight, ced_weight, latency_weight, fairness_weight]
    slack_sign = np.sign(slack)
    # When slack < 0: urgency gets 80%, others share 20% equally (CED 6.7%, latency 6.7%, fairness 6.7%)
    # When slack >= 0: urgency 40%, CED 30%, latency 20%, fairness 10%
    w_urgency = np.where(slack < 0, 0.8, 0.4)
    w_ced = np.where(slack < 0, 0.067, 0.3)
    w_latency = np.where(slack < 0, 0.067, 0.2)
    w_fairness = np.where(slack < 0, 0.066, 0.1)

    score = (
        w_urgency * urgency_term +
        w_ced * ced_term +
        w_latency * latency_term +
        w_fairness * fairness_term
    )

    # Final sanitization: ensure finite, deterministic, shape-(N,)
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    return score
