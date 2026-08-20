import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Novel priority rule emphasizing deadline risk containment, robust criticality gating,
    and balanced energy-duration tradeoff. Uses piecewise slack sensitivity and
    uncertainty-coupled urgency. Smaller score = higher priority."""
    eps = 1.6747014929403585e-09

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
    slack_urgency[slack < 0] = 4.643528379148883 * -slack[slack < 0]
    in_soft_region = (slack >= 0) & (slack <= slack_margin)
    slack_urgency[in_soft_region] = 0.503731828797202 * (slack[in_soft_region] / (slack_margin + eps)) ** 2
    norm_upward = robust_normalize(upward_rank)
    slack_penalty_factor = np.clip(1.0 - np.maximum(-slack, 0) / (slack_margin + eps), 0.0, 1.0)
    gated_criticality = norm_upward * slack_penalty_factor
    norm_uncertainty = robust_normalize(uncertainty)
    coupled_urgency = np.maximum(slack_urgency, 0) * np.clip(norm_uncertainty, 0.0, 2.0) * 0.3538066358789534
    wait_bias = 1.329036536025392 * ready_wait_time
    wait_bias_norm = robust_normalize(wait_bias)
    fairness_term = np.tanh(norm_duration) * norm_duration
    score = 4.665230071688514 * norm_energy + 0.694482465379491 * fairness_term - slack_urgency - coupled_urgency - 0.7672170331360646 * gated_criticality - wait_bias_norm
    score = np.nan_to_num(score, nan=np.median(score), posinf=np.max(score), neginf=np.min(score))
    return score
