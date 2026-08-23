import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """
    Self-evolved priority rule: replaces brittle threshold gates with smooth, bounded sigmoid slack gating;
    eliminates piecewise discontinuity and unstable early-reward terms; unifies urgency/criticality via monotonic risk scaling;
    retains robust normalization and finite safeguards.
    """
    eps = 5.451654077633291e-08

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
    sigmoid_input = 3.4117007890833566 * (norm_slack - -3.914571284745886)
    sigmoid_input_clipped = np.clip(sigmoid_input, -3.2225966454224992, 3.2225966454224992)
    slack_gate = 1.0 / (1.0 + np.exp(-sigmoid_input_clipped))
    urgency_base = np.maximum(0.0, -norm_slack)
    urgency_clipped = np.clip(urgency_base, 0.0, 3.2225966454224992)
    urgency_penalty = np.exp(3.4117007890833566 * urgency_clipped) * slack_gate
    critical_boost = 2.075051611853503 * norm_rank * (1.0 - slack_gate)
    deadline_pressure = np.maximum(0.0, -norm_slack)
    uncertainty_coupled_pressure = 0.9749023839313442 * deadline_pressure * norm_uncert * slack_gate
    wait_boost = 1.0 - np.exp(-0.3370250734147267 * norm_wait)
    rank_energy_penalty = 0.5422573526563619 * norm_energy * norm_rank
    score = urgency_penalty + critical_boost + uncertainty_coupled_pressure + rank_energy_penalty + 0.1289158127034845 * norm_energy + 0.983802712000096 * norm_duration - wait_boost
    finfo = np.finfo(np.float64)
    score = np.nan_to_num(score, nan=finfo.max, posinf=finfo.max, neginf=-finfo.max)
    return score
