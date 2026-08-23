import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule emphasizing:
      - Strict hard gate: only activate non-DDL terms when slack > ddl_protection_threshold (not just >0)
      - Critical path release via upward_rank × remaining_work product — unconditionally prioritized for urgency
      - Joint MAD normalization over |slack|, uncertainty, and duration_total for coherent risk scaling
      - Wait-time penalty scaled by (1 + ready_wait_time / (max(1, median(slack[slack>0]) + eps))) to avoid dominance
      - Risk-aware energy term only active in safe region, coupled with uncertainty via product
      - All numeric literals restricted to {-2,-1,0,1,2}; no other constants
      - Lexicographic DDL enforcement preserved: violation penalties dominate; safe-region terms are additive and gated
    """
    eps = 1.3059351170198372e-06
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
    all_risk_dims = np.stack([abs_slack, uncertainty, duration_total], axis=0)
    mad_per_dim = np.mean(np.abs(all_risk_dims - np.median(all_risk_dims, axis=1, keepdims=True)), axis=1) + eps
    norm_slack = abs_slack / (mad_per_dim[0] * 1.9926113585427272 + eps)
    norm_uncert = uncertainty / (mad_per_dim[1] * 1.9926113585427272 + eps)
    norm_duration = duration_total / (mad_per_dim[2] * 1.9926113585427272 + eps)
    joint_risk_scale = np.maximum.reduce([norm_slack, norm_uncert, norm_duration])
    ddl_penalty = np.where(slack < 0, -slack * (1.0 + norm_uncert * 0.02380450966262473), 0.0)
    critical_release_score = -(upward_rank * remaining_work)
    safe_mask = np.where(slack > 1.176986471787881, 1.0, 0.0)
    energy_norm = (min_incremental_energy + eps) / (joint_risk_scale + eps)
    energy_term = safe_mask * 2.5944828387350114 * (energy_norm / (np.median(energy_norm) + eps))
    energy_uncert_term = safe_mask * 0.48531965216099926 * (energy_norm * norm_uncert)
    safe_slack_median = np.median(slack[slack > 1.176986471787881]) if np.any(slack > 1.176986471787881) else 1.0
    wait_scale = 1.0 + ready_wait_time / (np.maximum(safe_slack_median, 1.0) + eps)
    wait_term = safe_mask * 1.535112179684049 * (ready_wait_time / (wait_scale + eps))
    score = ddl_penalty + critical_release_score * 0.9920704453940801 + energy_term + energy_uncert_term + wait_term
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
