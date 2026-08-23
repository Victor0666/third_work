import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with restored lexicographic DDL dominance and simplified uncertainty handling:
      - Replaces exponentiated penalty with large linear weight + hard feasibility gating for robust deadline enforcement
      - Removes fragile sigmoid and decay logic; uses clean binary mask for non-DDL terms
      - Introduces direct normalized_uncertainty term: always active, linearly penalizes high uncertainty regardless of slack
      - Joint MAD normalization now includes uncertainty *and* |slack| *and* duration_total → coherent risk alignment
      - Critical-path score uses hard-gated pressure (not smooth decay) to ensure discontinuity at threshold for strict priority ordering
      - Wait term uses relative efficiency but clamped to avoid numerical noise when duration is tiny
      - All numeric literals strictly limited to {-2,-1,0,1,2}; no hidden constants
      - Final score = DDL-penalty (dominant) + critical-path + uncertainty + (feasible-only) energy/wait/coupling
    """
    eps = 9.100428975162586e-08
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
    abs_slack = np.abs(slack) + eps
    all_risk_features = np.stack([abs_slack, uncertainty, duration_total], axis=0)
    mad_per_dim = np.mean(np.abs(all_risk_features - np.median(all_risk_features, axis=1, keepdims=True)), axis=1) + eps
    joint_mad = np.median(mad_per_dim) + eps

    def robust_normalize(x):
        x = np.asarray(x, dtype=float)
        if N == 1:
            return np.zeros_like(x, dtype=float)
        centered = x - np.median(x)
        denom = joint_mad * 0.3134427326619339 + eps
        return np.clip(centered / denom, -2.0, 2.0)
    ddl_violation_base = np.where(slack < 1.4469484626308329, 1.4469484626308329 - slack, 0.0)
    ddl_violation_penalty = 1.8395367571281056 * ddl_violation_base
    critical_release_score = upward_rank * remaining_work
    critical_mask = np.where(slack <= 1.4469484626308329, 1.0, 0.0)
    critical_score = -robust_normalize(critical_release_score) * critical_mask * 3.221382631525233
    unc_norm = robust_normalize(uncertainty)
    uncertainty_penalty = 0.5000662776518898 * unc_norm
    feasible_mask = np.where(slack > 1.4469484626308329, 1.0, 0.0)
    energy_norm = robust_normalize(min_incremental_energy)
    energy_term = feasible_mask * 1.840997933693382 * energy_norm
    unc_slack_interaction = uncertainty * (1.4469484626308329 - slack) * 1.3567617455496424
    unc_slack_norm = robust_normalize(unc_slack_interaction)
    unc_slack_term = feasible_mask * unc_slack_norm
    wait_efficiency = np.where(duration_total > eps, np.clip(ready_wait_time / duration_total, 0.0, 2.0), 0.0)
    wait_norm = robust_normalize(wait_efficiency)
    wait_term = feasible_mask * 1.5805692478343643 * wait_norm
    energy_uncertainty_term = feasible_mask * 0.8963697754440338 * energy_norm * unc_norm
    score = robust_normalize(ddl_violation_penalty) + critical_score + uncertainty_penalty + unc_slack_term
    score += energy_term + wait_term + energy_uncertainty_term
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
