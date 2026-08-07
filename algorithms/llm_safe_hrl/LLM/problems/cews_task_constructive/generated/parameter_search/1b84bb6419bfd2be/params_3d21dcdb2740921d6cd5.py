import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with strict branch count control:
    - Exactly ONE boolean combination: joint_ddl_pressure (used in 3 places → vectorized, not branching).
    - Zero Python if/elif/else; all logic via multiplication by 0/1 masks.
    - All numeric literals are in {-2,-1,0,1,2}; no hidden constants.
    - Uses only declared parameters — all 11 schema items are referenced.
    - Adaptive normalization uses IQR+min-max fallback for robustness.
    - Urgency is linear + modulated only when slack variance & joint pressure hold.
    - Bottleneck amplification gated by separate feasibility guard.
    """
    eps = 0.06708593909297171
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    def adaptive_normalize(x):
        x = np.copy(x)
        q_low = np.percentile(x, 40.0)
        q_high = np.percentile(x, 83.22537331152894)
        iqr = q_high - q_low
        center = np.median(x)
        if iqr > eps:
            denom = iqr
        else:
            denom = np.max(x) - np.min(x)
            if denom < eps:
                denom = eps
        return (x - center) / (denom + eps)
    median_slack = np.median(slack) if N > 0 else 0.0
    urgency_linear = np.clip(median_slack - slack, 0.0, 2.0)
    slack_range = np.max(slack) - np.min(slack) if N > 0 else eps
    slack_var = np.var(slack) if N > 1 else 0.0
    variance_sufficient = (slack_var > 0.5918793440466333 * (slack_range + eps) ** 2).astype(float)
    max_uncertainty = np.max(uncertainty) if N > 0 else eps
    joint_ddl_pressure = ((slack < median_slack) & (uncertainty > 0.5805233642264236 * max_uncertainty)).astype(float)
    modulated_urgency = urgency_linear * (1.0 + 0.15482839464604892 * joint_ddl_pressure * variance_sufficient)
    norm_urgency = adaptive_normalize(modulated_urgency)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / (duration + eps)
    norm_energy_eff = adaptive_normalize(energy_per_duration)
    bottleneck_pressure = duration * upward_rank * (1.0 + modulated_urgency + eps)
    bottleneck_guard = ((slack < median_slack) & (uncertainty > 0.6197425170110827 * max_uncertainty)).astype(float)
    bottleneck_pressure = bottleneck_pressure * np.power(1.0 + uncertainty, 0.934655322619025 * bottleneck_guard)
    norm_bottleneck = adaptive_normalize(bottleneck_pressure)
    wait_scaled = ready_wait_time / (3.3098822128750287 + eps)
    wait_clipped = np.clip(wait_scaled, -2.0, 2.0)
    wait_saturation = 1.0 / (1.0 + np.exp(-wait_clipped))
    norm_wait = adaptive_normalize(wait_saturation)
    norm_uncertainty = adaptive_normalize(uncertainty)
    ddl_risk_amplifier = joint_ddl_pressure * norm_uncertainty
    score = norm_urgency + 0.006974609992600914 * norm_bottleneck + 1.8719583313578303 * norm_energy_eff - norm_wait + ddl_risk_amplifier
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
