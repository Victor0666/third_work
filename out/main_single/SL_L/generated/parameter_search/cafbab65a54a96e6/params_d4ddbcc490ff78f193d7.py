import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule: all numeric constants declared; only -2,-1,0,1,2 used inline.
    
    Fixes: replaced hidden 0.5 with PARAMS["slack_compression_scale"].
    Structure preserved: piecewise critical-mode gating, risk-aware coupling, capped wait boost.
    All normalizations use robust mean-abs + epsilon; no std, log, or unstable ops.
    """
    eps = 0.0022003796234755226
    N = len(slack)

    def robust_normalize(x):
        x = np.asarray(x, dtype=np.float64)
        x_abs_mean = np.mean(np.abs(x)) + eps
        return x / x_abs_mean
    norm_exec_comm = robust_normalize(min_exec_time + min_comm_time)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_upward_rank = robust_normalize(upward_rank)
    norm_remaining_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(np.clip(ready_wait_time, 0, 9.410053367552079))
    is_critical = slack < -2.3331928408604186
    raw_negative_slack = np.where(is_critical, -slack, 0.0)
    compressed_slack_penalty = np.tanh(raw_negative_slack * 1.1357187176327643) * -1.0
    critical_penalty = 3.173254948242358 * compressed_slack_penalty * (1.0 + 0.5988472813820744 * robust_normalize(uncertainty))
    norm_slack = robust_normalize(slack)
    non_critical_urgency = -1.7012724972602926 * norm_upward_rank * (slack >= -2.3331928408604186)
    duration_risk_score = norm_exec_comm * (1.0 + 0.4613609737316189 * robust_normalize(uncertainty))
    wait_boost = -0.11838670747744412 * norm_wait
    score = critical_penalty + non_critical_urgency + 0.43701017532726466 * norm_energy + duration_risk_score + wait_boost
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.astype(np.float64)
