import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """
    Self-evolved priority rule: replaces brittle threshold gates with smooth, bounded sigmoid slack gating;
    eliminates piecewise discontinuity and unstable early-reward terms; unifies urgency/criticality via monotonic risk scaling;
    retains robust normalization and finite safeguards.
    """
    eps = 8.86542558003099e-08

    def robust_norm(x):
        x = np.asarray(x, dtype=np.float64)
        denom = np.mean(np.abs(x)) + eps
        return x / denom
    norm_slack = robust_norm(slack)
    norm_energy = robust_norm(min_incremental_energy)
    norm_duration = robust_norm(min_exec_time + min_comm_time)
    norm_rank = robust_norm(upward_rank)
    norm_wait = robust_norm(ready_wait_time)
    norm_uncert = robust_norm(uncertainty)
    sigmoid_input = 4.6026106700456335 * (norm_slack - -4.754646489392264)
    sigmoid_input_clipped = np.clip(sigmoid_input, -1.9927134145037522, 1.9927134145037522)
    slack_gate = 1.0 / (1.0 + np.exp(-sigmoid_input_clipped))
    urgency_base = np.maximum(0.0, -norm_slack)
    urgency_clipped = np.clip(urgency_base, 0.0, 1.9927134145037522)
    urgency_penalty = np.exp(4.6026106700456335 * urgency_clipped) * slack_gate
    critical_boost = 1.40354091978834 * norm_rank * (1.0 - slack_gate)
    deadline_pressure = np.maximum(0.0, -norm_slack)
    uncertainty_coupled_pressure = 0.24388433429772483 * deadline_pressure * norm_uncert * slack_gate
    wait_boost = 1.0 - np.exp(-0.1850091133960147 * norm_wait)
    rank_energy_penalty = 0.5874929318680694 * norm_energy * norm_rank
    score = urgency_penalty + critical_boost + uncertainty_coupled_pressure + rank_energy_penalty + 0.6144102464319072 * norm_energy + 0.3304856379357549 * norm_duration - wait_boost
    finfo = np.finfo(np.float64)
    score = np.nan_to_num(score, nan=finfo.max, posinf=finfo.max, neginf=-finfo.max)
    return score
