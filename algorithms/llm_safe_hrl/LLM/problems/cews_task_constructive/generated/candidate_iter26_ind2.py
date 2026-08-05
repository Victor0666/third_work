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
    v4 priority rule: Hard feasibility dominance + criticality-gated risk-energy density +
                      adaptive starvation rescue + uncertainty-calibrated slack sensitivity +
                      duration-aware criticality penalty.

    Key synthesis & innovations:
    - Retains Parent 2's hard feasibility dominance (violated tasks → -inf priority) for strict DDL adherence.
    - Integrates Parent 1's continuous criticality gating (sigmoid on upward_rank * rel_slack) to smoothly amplify urgency near deadlines.
    - Replaces fixed percentile starvation thresholds with dynamic criticality threshold (median + 0.5*IQR) AND wait-ratio > 2.0 AND slack < 300s.
    - Uses risk-normalized energy density: energy / (duration * (1+uncertainty)) — preserves fidelity under volatility, then weighted by upward_rank.
    - Introduces duration-aware criticality penalty: penalizes long tasks on high-criticality paths via (duration / max_duration) * upward_rank.
    - All normalization uses robust IQR-based z-clipping (±6σ) with tight bounds; no percentiles in core logic except for thresholding.
    - Adds uncertainty-calibrated slack sensitivity: exp(-max(0, -slack) / (duration + eps)) for sharp violation pressure + linear slack reward scaling.
    - Weights sum to 1.0: 0.45 (urgency) + 0.28 (risk-energy) + 0.15 (criticality-penalty) + 0.07 (starvation) + 0.05 (uncertainty).
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

    def robust_zclip(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.clip(x, -1e12, 1e12)
        if N == 1:
            return np.array([0.0])
        q25, q50, q75 = np.quantile(x, [0.25, 0.5, 0.75], method='midpoint')
        iqr = q75 - q25 + eps
        z = (x - q50) / iqr
        return np.clip(z, -6.0, 6.0)

    task_duration = np.maximum(min_exec_time + min_comm_time, eps)
    violated_mask = slack < 0
    # Hard feasibility dominance: assign extreme negative priority to violated tasks
    feasibility_boost = np.full(N, 0.0, dtype=np.float64)
    feasibility_boost[violated_mask] = -1e12

    # Criticality-gated risk-energy density: energy / (duration * (1+uncertainty))
    risk_adjusted_duration = task_duration * (1.0 + uncertainty)
    energy_density = np.divide(
        min_incremental_energy,
        risk_adjusted_duration,
        out=np.zeros_like(min_incremental_energy),
        where=risk_adjusted_duration != 0
    )
    energy_density = np.nan_to_num(energy_density, nan=0.0, posinf=0.0, neginf=0.0)

    # Relative slack for gating: avoid division by zero, clamp to [-10, 10]
    rel_slack = np.divide(slack, task_duration, out=np.zeros_like(slack), where=task_duration != 0)
    rel_slack = np.clip(rel_slack, -10.0, 10.0)
    # Criticality gate: sigmoid on upward_rank * clipped rel_slack (only positive slack contributes meaningfully)
    criticality_gate_input = upward_rank * np.clip(rel_slack, 0.0, 1.0)
    criticality_gate = 1.0 / (1.0 + np.exp(-(criticality_gate_input - 0.5)))

    # Uncertainty-calibrated slack sensitivity
    # Sharp exponential penalty for violations: exp(-max(0,-slack)/duration)
    violation_pressure = np.exp(-np.clip(-slack, 0.0, None) / (task_duration + eps))
    # Linear slack reward for non-violated tasks: normalized to [0,1] over median slack range
    if N > 1:
        q10, q50, q90 = np.quantile(slack, [0.1, 0.5, 0.9], method='midpoint')
        slack_reward = np.clip((q90 - slack) / (q90 - q10 + eps), 0.0, 1.0)
    else:
        slack_reward = np.ones_like(slack)
    slack_sensitivity = np.where(violated_mask, violation_pressure, slack_reward)

    # Dynamic starvation threshold: critical + waiting too long + still feasible
    if N > 1:
        ur_median = np.median(upward_rank)
        ur_q25, ur_q75 = np.quantile(upward_rank, [0.25, 0.75], method='midpoint')
        ur_iqr = ur_q75 - ur_q25 + eps
        crit_threshold = ur_median + 0.5 * ur_iqr
    else:
        crit_threshold = upward_rank[0]
    wait_ratio = np.divide(ready_wait_time, task_duration, out=np.zeros_like(ready_wait_time), where=task_duration != 0)
    wait_ratio = np.nan_to_num(wait_ratio, nan=0.0, posinf=0.0, neginf=0.0)
    is_starvable = (wait_ratio > 2.0) & (upward_rank >= crit_threshold) & (slack < 300.0)
    starvation_rescue = np.where(is_starvable, wait_ratio * (1.0 + 0.2 * np.clip(upward_rank, 0.0, 1.0)), 0.0)
    starvation_rescue = np.clip(starvation_rescue, 0.0, 3.0)

    # Duration-aware criticality penalty: penalize long tasks on critical paths
    max_duration = np.max(task_duration) + eps
    duration_penalty = (task_duration / max_duration) * upward_rank

    # Normalize components independently
    norm_urgency = robust_zclip(slack_sensitivity)
    norm_energy_density = robust_zclip(energy_density * criticality_gate)
    norm_penalty = robust_zclip(duration_penalty)
    norm_starvation = robust_zclip(starvation_rescue)
    norm_uncertainty = robust_zclip(uncertainty)

    # Weighted sum: sum-to-1 weights ensure scale invariance
    score = (
        0.45 * norm_urgency +
        0.28 * norm_energy_density +
        0.15 * norm_penalty +
        0.07 * norm_starvation +
        0.05 * norm_uncertainty
    )

    # Apply feasibility boost and clamp
    score = score + feasibility_boost
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    score = np.clip(score, -1e12, 1e12)

    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
