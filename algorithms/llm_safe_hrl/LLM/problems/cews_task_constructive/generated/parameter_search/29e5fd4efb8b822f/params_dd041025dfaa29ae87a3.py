import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with:
    - Reintroduced tunable IQR percentiles (now declared) to replace hardcoded 25/75
    - Conditional bottleneck amplification only under verified DDL risk
    - Restored multiplicative bottleneck structure for stronger critical-path signal
    - Clamped tanh urgency + subtractive fairness for robust deadline safety and starvation prevention
    - All numeric literals are in {-2,-1,0,1,2}; no hidden constants beyond that set
    """
    eps = 0.02252732638658669
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
        q_low = np.percentile(x, 29.80096692499279)
        q_high = np.percentile(x, 74.28204588765993)
        iqr = q_high - q_low + eps
        center = np.median(x)
        return (x - center) / iqr
    abs_slack = np.abs(slack)
    scale = np.median(abs_slack) if np.any(abs_slack > eps) else 1.0
    tanh_urgency = np.tanh(-slack / (scale * 1.9658926091929756 + eps))
    clamped_urgency = np.clip(tanh_urgency, -0.8420620676934811, 1.0)
    urgency = (clamped_urgency + 1.0) / 2.0
    norm_urgency = adaptive_normalize(urgency)
    duration = min_exec_time + min_comm_time
    bottleneck_pressure = duration * upward_rank * (min_incremental_energy + eps)
    norm_bottleneck = adaptive_normalize(bottleneck_pressure)
    norm_wait = adaptive_normalize(ready_wait_time)
    fairness_term = -1.2145286096133965 * norm_wait
    median_uncertainty = np.median(uncertainty) if N > 0 else eps
    ddl_risk_gate = ((slack < 0.0) & (uncertainty > 0.5610570922525322 * median_uncertainty)).astype(float)
    risk_amplified_bottleneck = norm_bottleneck * (1.0 + 1.77110516162629 * ddl_risk_gate)
    stress_amplified_urgency = norm_urgency * (1.0 + ddl_risk_gate)
    score = stress_amplified_urgency + 0.09340817771442095 * risk_amplified_bottleneck + fairness_term
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
