import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Novel priority rule emphasizing deadline risk containment, robust criticality gating,
    and balanced energy-duration tradeoff. Uses piecewise slack sensitivity and
    uncertainty-coupled urgency. Smaller score = higher priority."""
    eps = 1.631933578906869e-08

    def robust_normalize(x):
        x = np.asarray(x, dtype=float)
        center = np.median(x)
        scale = np.median(np.abs(x - center)) + eps
        return (x - center) / scale
    duration = min_exec_time + min_comm_time
    norm_duration = robust_normalize(duration)
    norm_energy = robust_normalize(min_incremental_energy)
    slack = np.asarray(slack, dtype=float)
    slack_abs = np.abs(slack)
    pos_slack = slack[slack > eps]
    slack_margin = np.median(pos_slack) if len(pos_slack) > 0 else 1.0
    slack_margin = max(slack_margin, eps)
    slack_urgency = np.zeros_like(slack)
    slack_urgency[slack < 0] = 9.353458146736806 * -slack[slack < 0]
    in_soft_region = (slack >= 0) & (slack <= slack_margin)
    slack_urgency[in_soft_region] = 5.843795900879908 * (slack[in_soft_region] / (slack_margin + eps)) ** 2
    norm_upward = robust_normalize(upward_rank)
    slack_penalty_factor = np.clip(1.0 - np.maximum(-slack, 0) / (slack_margin + eps), 0.0, 1.0)
    gated_criticality = norm_upward * slack_penalty_factor
    norm_uncertainty = robust_normalize(uncertainty)
    coupled_urgency = np.maximum(slack_urgency, 0) * np.clip(norm_uncertainty, 0.0, 2.0) * 1.2104387277086608
    wait_bias = 0.538429228832697 * ready_wait_time
    wait_bias_norm = robust_normalize(wait_bias)
    fairness_term = np.tanh(norm_duration) * norm_duration
    score = 1.5924941306370048 * norm_energy + 0.683857321077346 * fairness_term - slack_urgency - coupled_urgency - 0.5204648815756718 * gated_criticality - wait_bias_norm
    score = np.nan_to_num(score, nan=np.median(score), posinf=np.max(score), neginf=np.min(score))
    return score
