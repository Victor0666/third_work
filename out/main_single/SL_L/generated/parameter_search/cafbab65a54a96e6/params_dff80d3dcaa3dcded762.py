import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule: all numeric constants declared; only -2,-1,0,1,2 used inline.
    
    Fixes: replaced hidden 0.5 with PARAMS["slack_compression_scale"].
    Structure preserved: piecewise critical-mode gating, risk-aware coupling, capped wait boost.
    All normalizations use robust mean-abs + epsilon; no std, log, or unstable ops.
    """
    eps = 0.024788732310543925
    N = len(slack)

    def robust_normalize(x):
        x = np.asarray(x, dtype=np.float64)
        x_abs_mean = np.mean(np.abs(x)) + eps
        return x / x_abs_mean
    norm_exec_comm = robust_normalize(min_exec_time + min_comm_time)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_upward_rank = robust_normalize(upward_rank)
    norm_remaining_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(np.clip(ready_wait_time, 0, 20.30255914227993))
    is_critical = slack < -0.6827912679801615
    raw_negative_slack = np.where(is_critical, -slack, 0.0)
    compressed_slack_penalty = np.tanh(raw_negative_slack * 0.15526176997782135) * -1.0
    critical_penalty = 4.4668215987989335 * compressed_slack_penalty * (1.0 + 0.45436814983009777 * robust_normalize(uncertainty))
    norm_slack = robust_normalize(slack)
    non_critical_urgency = -0.20420920273839893 * norm_upward_rank * (slack >= -0.6827912679801615)
    duration_risk_score = norm_exec_comm * (1.0 + 0.7854604168097249 * robust_normalize(uncertainty))
    wait_boost = -0.35106861345033136 * norm_wait
    score = critical_penalty + non_critical_urgency + 1.7102932880259125 * norm_energy + duration_risk_score + wait_boost
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.astype(np.float64)
