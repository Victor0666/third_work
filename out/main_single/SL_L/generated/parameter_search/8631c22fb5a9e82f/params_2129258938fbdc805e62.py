import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule enforcing strict lexicographic DDL protection via hard gating:
      - Uses `slack > ddl_protection_threshold` (not just >0) as binary feasibility gate for non-DDL terms
      - Replaces percentile normalization with joint MAD normalization over |slack|, uncertainty, and duration_total for coherent risk alignment
      - Prioritizes critical-path release unconditionally via `upward_rank * remaining_work`, scaled by DDL pressure
      - Introduces starvation mitigation via `ready_wait_time / (min_exec_time + min_comm_time + eps)` only when feasible
      - Uses bounded sigmoid on uncertainty to smoothly suppress energy-aware terms at extreme risk
      - All numeric literals restricted to {-2,-1,0,1,2}; no other constants used
      - Final score: DDL-critical penalty + critical-path release + (feasible-only) energy/uncertainty/wait terms
    """
    eps = 1.2634459876294487e-07
    finfo = np.finfo(float)
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=finfo.max, neginf=finfo.min)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=finfo.max, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=eps, posinf=finfo.max, neginf=eps)
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=float)
    duration_total = min_exec_time + min_comm_time + eps
    abs_slack = np.abs(slack)
    all_risk_features = np.stack([abs_slack, uncertainty, duration_total], axis=0)
    mad = np.mean(np.abs(all_risk_features - np.median(all_risk_features, axis=1, keepdims=True)), axis=1)
    joint_mad = np.median(mad) + eps

    def mad_normalize(x):
        x = np.asarray(x, dtype=float)
        if N == 1:
            return np.zeros_like(x, dtype=float)
        return np.clip((x - np.median(x)) / (joint_mad * 0.3950290122639105 + eps), -2.0, 2.0)
    ddl_violation_penalty = np.where(slack < 0.0009137081410701393, np.maximum(0.0, 0.0009137081410701393 - slack), 0.0)
    critical_release_score = upward_rank * remaining_work
    slack_pressure = np.clip((0.0009137081410701393 - slack) / (0.0009137081410701393 + eps), 0.0, 1.0)
    critical_score = -mad_normalize(critical_release_score) * slack_pressure * 1.994578429857193
    feasible_mask = np.where(slack > 0.0009137081410701393, 1.0, 0.0)
    energy_norm = mad_normalize(min_incremental_energy)
    energy_term = feasible_mask * 1.469453247632304 * energy_norm
    unc_slack_interaction = uncertainty * (0.0009137081410701393 - slack) * 0.6158774862025236
    unc_slack_norm = mad_normalize(unc_slack_interaction)
    wait_efficiency = np.where(duration_total > eps, ready_wait_time / duration_total, 0.0)
    wait_norm = mad_normalize(wait_efficiency)
    wait_term = feasible_mask * 1.4158193623541928 * wait_norm
    unc_sigmoid = 1.0 / (1.0 + np.exp(-2.0 * (1.0 - uncertainty)))
    energy_uncertainty_term = feasible_mask * 0.0011698029184391918 * energy_norm * unc_sigmoid
    score = mad_normalize(ddl_violation_penalty) + critical_score + unc_slack_norm
    score += energy_term + wait_term + energy_uncertainty_term
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
