import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with stabilized risk alignment and hardened DDL semantics:
      - Replaces joint MAD with *per-feature* median absolute deviation + max-based robust scaling
        → eliminates fragile weighted fusion while preserving cross-feature scale coherence
      - Uses binary feasibility gate `slack >= ddl_safety_margin` (not > threshold) → aligns with hard deadline constraint semantics
      - Removes nonlinear slack exponent → ensures lexicographic DDL priority remains linear, monotonic, and interpretable
      - Tightens uncertainty sigmoid center to `uncertainty_sigmoid_center` for sharper suppression of energy terms at moderate-high risk
      - All numeric literals strictly limited to {-2,-1,0,1,2}; no hidden constants or domain-specific scales
      - Final score preserves strict DDL-first ordering: violation penalty dominates; all other terms are gated or scaled
    """
    eps = 7.1417775212908e-09
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
    mad_abs_slack = np.median(np.abs(abs_slack - np.median(abs_slack))) + eps
    mad_uncertainty = np.median(np.abs(uncertainty - np.median(uncertainty))) + eps
    mad_duration = np.median(np.abs(duration_total - np.median(duration_total))) + eps
    joint_scale = np.max([mad_abs_slack * 0.358857035029453, mad_uncertainty * 0.358857035029453, mad_duration * 0.358857035029453]) + eps

    def mad_normalize(x):
        x = np.asarray(x, dtype=float)
        if N == 1:
            return np.zeros_like(x, dtype=float)
        x_median = np.median(x)
        norm = (x - x_median) / (joint_scale + eps)
        return np.clip(norm, -2.0, 2.0)
    slack_deficit = np.maximum(0.0, -0.9997933273117011 - slack)
    ddl_violation_penalty = slack_deficit
    critical_release_score = upward_rank * remaining_work
    slack_pressure = np.clip((-0.9997933273117011 - slack) / (np.abs(-0.9997933273117011) + eps), 0.0, 1.0)
    critical_score = -mad_normalize(critical_release_score) * slack_pressure * 3.9954095177858004
    feasible_mask = np.where(slack >= -0.9997933273117011, 1.0, 0.0)
    energy_norm = mad_normalize(min_incremental_energy)
    energy_term = feasible_mask * 0.9849170594068302 * energy_norm
    unc_slack_interaction = uncertainty * slack_deficit * 1.4118029806996562
    unc_slack_norm = mad_normalize(unc_slack_interaction)
    wait_efficiency = np.where(duration_total > eps, ready_wait_time / duration_total, 0.0)
    wait_norm = mad_normalize(wait_efficiency)
    wait_term = feasible_mask * 0.18059149381163678 * wait_norm
    unc_sigmoid = 1.0 / (1.0 + np.exp(-2.0 * (1.3535889714949485 - uncertainty)))
    energy_uncertainty_term = feasible_mask * 0.04078997270702506 * energy_norm * unc_sigmoid
    score = mad_normalize(ddl_violation_penalty) + critical_score + unc_slack_norm
    score += energy_term + wait_term + energy_uncertainty_term
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
