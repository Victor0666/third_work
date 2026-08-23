import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """
    Self-evolved priority rule: replaces brittle threshold gates with smooth, bounded sigmoid slack gating;
    eliminates piecewise discontinuity and unstable early-reward terms; unifies urgency/criticality via monotonic risk scaling;
    retains robust normalization and finite safeguards.
    """
    eps = 4.990787603810944e-07

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
    sigmoid_input = 2.371154095903832 * (norm_slack - -4.625612785043228)
    sigmoid_input_clipped = np.clip(sigmoid_input, -2.8375943687558562, 2.8375943687558562)
    slack_gate = 1.0 / (1.0 + np.exp(-sigmoid_input_clipped))
    urgency_base = np.maximum(0.0, -norm_slack)
    urgency_clipped = np.clip(urgency_base, 0.0, 2.8375943687558562)
    urgency_penalty = np.exp(2.371154095903832 * urgency_clipped) * slack_gate
    critical_boost = 2.093127305957129 * norm_rank * (1.0 - slack_gate)
    deadline_pressure = np.maximum(0.0, -norm_slack)
    uncertainty_coupled_pressure = 1.191536747890661 * deadline_pressure * norm_uncert * slack_gate
    wait_boost = 1.0 - np.exp(-0.14832550741871675 * norm_wait)
    rank_energy_penalty = 0.5658429011083048 * norm_energy * norm_rank
    score = urgency_penalty + critical_boost + uncertainty_coupled_pressure + rank_energy_penalty + 2.990803659152624 * norm_energy + 0.785465118703482 * norm_duration - wait_boost
    finfo = np.finfo(np.float64)
    score = np.nan_to_num(score, nan=finfo.max, posinf=finfo.max, neginf=-finfo.max)
    return score
