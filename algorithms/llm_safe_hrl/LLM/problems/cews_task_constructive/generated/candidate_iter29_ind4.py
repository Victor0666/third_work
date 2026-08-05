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
    v2 priority rule: Hybrid urgency-criticality-energy-fairness with robust risk-gating.
    
    Combines Parent 2's hard-deadline enforcement and adaptive slack scaling with
    Parent 1's smooth tanh-based urgency and starvation-resilient wait rescue.
    Key novelties:
    - Unified urgency base: tanh(-rel_slack) for smooth gradient + hard lateness penalty
    - Critical timing gated by both slack proximity (tanh(|rel_slack|)) AND upward_rank
    - Energy penalty uses dual gating: (upward_rank > median) * (slack > 0) * tanh(-rel_slack)
    - Fairness boost now work-density-normalized and activated only under positive slack
    - Uncertainty coupling requires slack > 0 AND upward_rank > median AND uncertainty > median
    - All components normalized via robust 5–95% clipping (more stable than 1–99% for small N)
    - Final score bounded in [-1e12, 1e12] with deterministic fallbacks
    """
    eps = 1e-08
    min_exec_time = np.asarray(min_exec_time, dtype=np.float64).copy()
    min_comm_time = np.asarray(min_comm_time, dtype=np.float64).copy()
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=np.float64).copy()
    slack = np.asarray(slack, dtype=np.float64).copy()
    upward_rank = np.asarray(upward_rank, dtype=np.float64).copy()
    remaining_work = np.asarray(remaining_work, dtype=np.float64).copy()
    ready_wait_time = np.asarray(ready_wait_time, dtype=np.float64).copy()
    uncertainty = np.asarray(uncertainty, dtype=np.float64).copy()
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=np.float64)

    def robust_5_95_norm(x):
        if x.size == 0:
            return np.zeros_like(x)
        if N == 1:
            return np.zeros_like(x)
        p05 = np.percentile(x, 5.0, method='lower')
        p95 = np.percentile(x, 95.0, method='higher')
        x_clipped = np.clip(x, p05, p95)
        x_min = np.min(x_clipped)
        x_max = np.max(x_clipped)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        return (x_clipped - x_min) / (x_max - x_min + eps)

    # Lateness penalty: hard enforcement with smooth fallback
    lateness_mask = slack <= 0.0
    abs_slack = np.abs(slack)
    lateness_penalty = np.where(lateness_mask, -1e12 * (1.0 + 0.1 * abs_slack), 0.0)

    # Duration and relative slack
    duration = min_exec_time + min_comm_time + eps
    rel_slack = np.divide(slack, duration, out=np.zeros_like(slack), where=duration != 0)
    rel_slack = np.nan_to_num(rel_slack, nan=0.0, posinf=0.0, neginf=0.0)

    # Smooth urgency base: tanh(-rel_slack) → [0,1], 1 when urgent, 0 when slack abundant
    urgency_base = np.tanh(-rel_slack)

    # Critical timing: duration * upward_rank, scaled by urgency to avoid premature scheduling
    critical_timing = duration * upward_rank
    norm_critical_timing = robust_5_95_norm(critical_timing)
    # Slack-scaled criticality: suppress when slack is tight but positive
    slack_scale = np.clip(1.0 - np.abs(rel_slack), 0.0, 1.0)
    scaled_critical_timing = norm_critical_timing * slack_scale

    # Energy density and penalty: only penalize high-energy tasks when critical and urgent
    energy_density = np.divide(min_incremental_energy, duration, out=np.zeros_like(min_incremental_energy), where=duration != 0)
    energy_density = np.nan_to_num(energy_density, nan=0.0, posinf=0.0, neginf=0.0)
    norm_energy_density = robust_5_95_norm(energy_density)
    # Dual gating: only apply energy penalty to critical (upward_rank > median) and urgent (slack > 0) tasks
    ur_median = np.median(upward_rank) if N > 1 else upward_rank[0]
    energy_gate = (upward_rank > ur_median) & (slack > 0.0)
    energy_penalty = norm_energy_density * urgency_base * energy_gate

    # Fairness boost: rescue starved high-work tasks only when deadline allows
    rw_median = np.median(remaining_work) if N > 1 else remaining_work[0]
    fairness_mask = (slack > 0.0) & (remaining_work > rw_median)
    # Normalize wait time by work density to prioritize waiting on heavy tasks
    work_density = np.divide(remaining_work, duration + eps, out=np.zeros_like(remaining_work), where=duration + eps != 0)
    wait_score = np.divide(ready_wait_time, work_density + eps, out=np.zeros_like(ready_wait_time), where=work_density + eps != 0)
    norm_wait_score = robust_5_95_norm(wait_score)
    fairness_boost = norm_wait_score * fairness_mask

    # Uncertainty coupling: only amplify risk-awareness when all three conditions hold
    unc_median = np.median(uncertainty) if N > 1 else uncertainty[0]
    unc_mask = (slack > 0.0) & (upward_rank > ur_median) & (uncertainty > unc_median)
    norm_uncertainty = robust_5_95_norm(uncertainty)
    unc_coupling = norm_uncertainty * unc_mask

    # Final weighted score: urgency dominates, criticality second, energy and fairness follow
    score = (
        0.42 * scaled_critical_timing +
        0.23 * urgency_base +
        0.17 * energy_penalty +
        0.10 * fairness_boost +
        0.05 * unc_coupling +
        0.03 * robust_5_95_norm(remaining_work)
    )
    score = lateness_penalty + score

    # Ensure determinism and numerical safety
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    score = np.clip(score, -1e12, 1e12)

    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
