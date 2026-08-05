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
    # Robust input sanitization: handle NaN, inf, and extreme values upfront
    eps = 1e-08
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=0.0, posinf=1e9, neginf=eps)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=0.0, posinf=1e9, neginf=eps)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=0.0, posinf=1e9, neginf=eps)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=1e9, neginf=-1e9)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=0.0, posinf=1e9, neginf=eps)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=0.0, posinf=1e9, neginf=eps)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=1e9, neginf=eps)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=0.0, posinf=1e9, neginf=eps)

    N = len(slack)

    # Rank-based standardization: robust to N=1, outliers, flat arrays; avoids median/IQR artifacts
    def rank_standardize(x):
        if N == 1:
            return np.array([0.0])
        ranks = np.argsort(np.argsort(x)) + 1.0  # 1-based ranks
        mean_rank = np.mean(ranks)
        std_rank = np.std(ranks) if N > 1 else 1.0
        return (ranks - mean_rank) / (std_rank + eps)

    # === Deadline Urgency (DDL-hardness first) ===
    # Asymmetric urgency: steeper response for negative slack (violation risk), gentler for tight-but-safe
    asym_factor = np.where(slack < 0, 3.0, 0.7)
    tau = np.abs(np.median(slack)) + eps if N > 1 else 1.0
    urgency_sigmoid = 1.0 / (1.0 + np.exp(-asym_factor * slack / tau))
    # Hard penalty only when violation is imminent (slack < -eps)
    hard_violation_penalty = np.where(slack < -eps, -slack * 5.0, 0.0)
    deadline_urgency = urgency_sigmoid + hard_violation_penalty
    norm_urgency = rank_standardize(deadline_urgency)
    urgency_term = -3.0 * norm_urgency  # Higher priority for urgent tasks → lower score

    # === Criticality-Gated Energy Efficiency ===
    # Prioritize energy reduction *only* when slack permits; otherwise defer efficiency optimization
    exec_effort = np.maximum(min_exec_time, eps)
    critical_density = upward_rank * (remaining_work / exec_effort)
    # Gated efficiency: energy-per-critical-unit penalized only under violation
    latency_cost = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_latency = min_incremental_energy / latency_cost
    eff_per_crit = energy_per_latency / (critical_density + eps)
    eff_penalty_mask = (slack < 0).astype(float)
    gated_eff_score = eff_per_crit * eff_penalty_mask + np.mean(eff_per_crit + eps) * (1.0 - eff_penalty_mask)
    norm_eff = rank_standardize(gated_eff_score)
    efficiency_term = -0.8 * norm_eff

    # === Starvation-Resilient Aging Boost ===
    # Relative wait quantile prevents skew bias; scaled by urgency to avoid overriding deadlines
    if N == 1:
        rel_wait_quantile = np.array([0.5])
    else:
        sorted_wait = np.sort(ready_wait_time)
        # Use linear interpolation for robust quantile estimation
        rel_wait_quantile = (np.searchsorted(sorted_wait, ready_wait_time, side='right') + 0.5) / N
    aging_boost = urgency_sigmoid * rel_wait_quantile
    norm_aging = rank_standardize(aging_boost)
    aging_term = -0.4 * norm_aging

    # === Uncertainty-Aware Risk Amplification ===
    # Amplify uncertainty impact near zero slack, saturate gracefully for large |slack|
    abs_slack = np.abs(slack) + eps
    risk_amplifier = 1.0 + (abs_slack / (abs_slack + eps))  # ∈ [1.0, 2.0]
    risk_amplified_uncertainty = uncertainty * risk_amplifier
    # Penalty strength increases with violation depth
    abs_slack_violation = np.maximum(-slack, 0.0)
    uncertainty_penalty = risk_amplified_uncertainty * abs_slack_violation
    norm_uncertainty = rank_standardize(uncertainty_penalty)
    uncertainty_term = 0.5 * norm_uncertainty

    # === Critical Path Leverage Term (from Parent 1 insight) ===
    # Encourage early scheduling of high-upward-rank tasks *when not violating DDL*
    # Scaled by slack safety margin to avoid interfering with urgency
    slack_safety = np.clip(slack / (np.abs(np.median(slack)) + eps), 0.0, 1.0)
    critical_leverage = upward_rank * slack_safety
    norm_critical_leverage = rank_standardize(critical_leverage)
    critical_leverage_term = -0.6 * norm_critical_leverage

    # Aggregate components — all terms designed so lower score = higher priority
    score = (
        urgency_term +
        efficiency_term +
        aging_term +
        uncertainty_term +
        critical_leverage_term
    )

    # Final robustness: eliminate NaN/inf, ensure finite output
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)

    # Ensure shape (N,) even in edge cases (e.g., constant inputs)
    if score.shape != (N,):
        score = np.broadcast_to(score, (N,))

    return score
