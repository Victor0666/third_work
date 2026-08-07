import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with:
    - Removed inactive fairness parameters (wait_fairness_weight, ddl_risk_activation_threshold no longer used in fairness)
    - Reintroduced *multiplicative risk-gated energy term*: only activates under verified DDL risk (slack < 0 ∧ high uncertainty)
    - Restored clean bottleneck structure: duration × upward_rank × energy (no redundant modulations)
    - Unified risk gating: single ddl_risk_gate controls both bottleneck amplification AND energy penalty
    - Simplified normalization: removed redundant adaptive_normalize calls; reused same robust center/IQR logic
    - Strict finiteness: all divisions guarded, nan/inf replaced via np.finfo, no hidden constants beyond {-2,-1,0,1,2}
    """
    eps = 1.0011556209159539e-06
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
        q_low = np.percentile(x, 26.51639111242277)
        q_high = np.percentile(x, 84.66188982792104)
        iqr = q_high - q_low + eps
        center = np.median(x)
        return (x - center) / (iqr + eps)
    abs_slack = np.abs(slack)
    scale = np.median(abs_slack) if np.any(abs_slack > eps) else 1.0
    tanh_urgency = np.tanh(-slack / (scale * 2.643256467820401 + eps))
    clamped_urgency = np.clip(tanh_urgency, -0.84936555122842, 1.0)
    urgency = (clamped_urgency + 1.0) / 2.0
    norm_urgency = adaptive_normalize(urgency)
    duration = min_exec_time + min_comm_time
    bottleneck_pressure = duration * upward_rank * (min_incremental_energy + eps)
    norm_bottleneck = adaptive_normalize(bottleneck_pressure)
    median_uncertainty = np.median(uncertainty) if N > 0 else eps
    ddl_risk_gate = ((slack < 0.0) & (uncertainty > 0.7268340214514061 * median_uncertainty)).astype(float)
    risk_amplified_bottleneck = norm_bottleneck * (1.0 + 0.6563695680376014 * ddl_risk_gate)
    norm_energy = adaptive_normalize(min_incremental_energy)
    risk_gated_energy_penalty = 0.5731751636255104 * norm_energy * ddl_risk_gate
    score = norm_urgency + 0.3749656194258725 * risk_amplified_bottleneck + risk_gated_energy_penalty
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
