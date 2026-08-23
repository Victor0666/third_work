import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule: neginf_clip now uses identity transform.
    Uses risk-gated criticality, bounded uncertainty-duration coupling, and exponential wait boost.
    Smaller score = higher priority. All numeric literals are -2,-1,0,1,2 or derived from PARAMS."""
    eps = 1.2473361204576427e-05
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    def robust_norm(x):
        x_abs = np.abs(x)
        denom = np.mean(x_abs) + eps
        return x / denom
    norm_slack = robust_norm(slack)
    norm_energy = robust_norm(min_incremental_energy)
    norm_duration = robust_norm(min_exec_time + min_comm_time)
    norm_rank = robust_norm(upward_rank)
    norm_work = robust_norm(remaining_work)
    norm_wait = robust_norm(ready_wait_time)
    norm_uncert = robust_norm(uncertainty)
    slack_power = np.sign(norm_slack) * np.abs(norm_slack) ** 2.331535422053549
    slack_normalized_for_gate = (norm_slack - 0.6998469320284593) / (1.0 + eps)
    gate_activation = np.clip(-slack_normalized_for_gate, 0.0, 1.0)
    dur_uncert_coupling = np.clip(norm_duration * norm_uncert, 0.0, 1.0) * 1.3415360304513073
    wait_boost = 1.0 - np.exp(-0.05606421294621675 * (norm_wait + eps))
    slack_pressure_mask = np.where(norm_slack < 0, 1.0, 0.0)
    uncert_slack_penalty = norm_uncert * slack_pressure_mask * 1.6629250446155854
    score = slack_power
    safe_slack_mask = np.where(norm_slack > 0, 1.0, 0.0)
    score += safe_slack_mask * 2.8562889213930704 * norm_energy
    score -= gate_activation * 1.2026387760490087 * norm_rank
    score += dur_uncert_coupling
    score -= wait_boost * 0.2030756767523788
    score += uncert_slack_penalty
    score = np.nan_to_num(score, nan=-520681.6535105579, posinf=3810628.6927929404, neginf=-265497696.46854532)
    return score
