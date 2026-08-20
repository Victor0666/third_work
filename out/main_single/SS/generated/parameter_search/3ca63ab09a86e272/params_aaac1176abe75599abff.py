import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: removes destabilizing successor_release_score; restores monotonic clipped-power slack pressure; 
       introduces negative-slack-only activation for uncertainty-duration penalty to enforce hard deadline feasibility;
       retains robust median/MAD normalization, smooth uncertainty sigmoid, restored rank-slack coupling, and saturating wait relief."""
    eps = 0.000139511391530366
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def robust_normalize(x):
        if N == 0:
            return np.zeros(0, dtype=float)
        x_abs = np.abs(x)
        center = np.median(x_abs)
        dev = x_abs - center
        scale = np.median(np.abs(dev)) + eps
        return dev / (scale + eps)
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    slack_pressure = np.power(np.clip(-norm_slack, 0.0, None), 1.5839205879041873)
    rank_gate = np.clip(1.0 - norm_slack / (0.0850801157707337 + eps), 0.0, 1.0)
    boosted_rank = norm_rank * (1.0 + 1.3946858399622715 * rank_gate)
    uncertainty_activation = 1.0 / (1.0 + np.exp(-0.7251124362845496 * (norm_uncert - 0.06537064722126436)))
    negative_slack_mask = (slack < 0.0).astype(float) * 0.9343747683740953
    duration_risk_interaction = norm_duration * uncertainty_activation * 0.04551802963023106 * negative_slack_mask
    wait_benefit = 1.0 - np.exp(-0.05092128295126852 * (norm_wait + eps))
    energy_slack_penalty = norm_energy * (1.0 + 0.6145534709454004 * slack_pressure)
    coupled_rank_reward = norm_rank * (1.0 + 0.6309244228519797 * slack_pressure)
    score = +slack_pressure + energy_slack_penalty + 1.0252570064270543 * norm_energy - coupled_rank_reward - wait_benefit + duration_risk_interaction
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
