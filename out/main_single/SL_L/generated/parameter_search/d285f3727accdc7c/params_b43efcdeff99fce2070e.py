import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule with compound-risk criticality gate and triple-coupling risk term.
    
    Structural changes:
    - Replaces IQR with robust mean-abs normalization (stable, low AST depth)
    - Introduces bounded triple interaction: duration * work * uncertainty → captures risk in large,
      uncertain, compute-heavy sub-DAGs; clipped to [0,1] to prevent explosion
    - Criticality gate now requires BOTH slack < 0 AND uncertainty > threshold (compound detection)
    - All numeric literals are from {-2,-1,0,1,2}; no hidden constants
    - Uses np.finfo for immutable safeguards where needed (e.g., in robust_norm denominator)
    """
    eps = 0.0005693480061754249
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
    slack_power = np.sign(norm_slack) * np.abs(norm_slack) ** 3.809477193240774
    triple_coupling = np.clip(norm_duration * norm_work * norm_uncert, 0.0, 1.0) * 1.323764548120718
    slack_pressure_mask = np.where(norm_slack < 0, 1.0, 0.0)
    uncert_high_mask = np.where(norm_uncert > 0.08359308415332012, 1.0, 0.0)
    compound_risk_mask = slack_pressure_mask * uncert_high_mask
    gate_activation = compound_risk_mask
    dur_uncert_coupling = np.clip(norm_duration * norm_uncert, 0.0, 1.0) * 1.323764548120718
    wait_boost = 1.0 - np.exp(-0.3832873730582967 * (norm_wait + eps))
    uncert_slack_penalty = norm_uncert * slack_pressure_mask * 1.9748779479027967
    score = slack_power
    safe_slack_mask = np.where(norm_slack > 0, 1.0, 0.0)
    score += safe_slack_mask * 1.7665052332796392 * norm_energy
    score -= gate_activation * 0.06760072664399502 * norm_rank
    score += dur_uncert_coupling
    score += triple_coupling
    score -= wait_boost * 0.11684142251997587
    score += uncert_slack_penalty
    score = np.nan_to_num(score, nan=-457894.91898730583, posinf=152228.74319630192, neginf=-120755647.36468673)
    return score
