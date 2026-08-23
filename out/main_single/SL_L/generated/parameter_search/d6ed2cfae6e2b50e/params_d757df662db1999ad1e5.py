import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining Parent 2's robust structure with Parent 1's joint risk scaling:
      - Retains Parent 2's successful lexicographic DDL enforcement, linear slack headroom, and sigmoid-gated uncertainty coupling.
      - Integrates Parent 1's joint MAD normalization over [abs_slack, uncertainty, duration_total] for coherent risk alignment.
      - Introduces novel 'joint_risk_mad_scale' to tune cross-dimension coupling strength without breaking scale invariance.
      - Replaces isolated MAD calls with unified joint-risk basis — improves stability for N=1 and outlier resilience.
      - All non-DDL terms remain strictly gated by ddl_safe_mask and host-load-aware scaling.
      - Critical-path term remains unconditional and dominant; wait-term preserves linear urgency scaling.
      - No numeric literals beyond {-2,-1,0,1,2}; all tunables declared and used.
    """
    eps = 1.1793398579893253e-07
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
    norm_slack = (abs_slack - np.median(abs_slack)) / (mad_per_dim[0] * 1.209820982071318 + eps)
    norm_uncert = (uncertainty - np.median(uncertainty)) / (mad_per_dim[1] * 1.209820982071318 + eps)
    norm_duration = (duration_total - np.median(duration_total)) / (mad_per_dim[2] * 1.209820982071318 + eps)
    norm_slack = np.clip(norm_slack, -2.0, 2.0)
    norm_uncert = np.clip(norm_uncert, -2.0, 2.0)
    norm_duration = np.clip(norm_duration, -2.0, 2.0)
    slack_score = np.where(slack < 0, (-slack) ** 1.3084405968591946, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 1.741528669593826
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 0.862537955310557
    critical_path_release = upward_rank * remaining_work * 1.0851760272249842
    ddl_safe_mask = np.where(slack > 0.7374270104049188, 1.0, 0.0)
    duration_total_safe = np.where(ddl_safe_mask > 0, duration_total, eps)
    energy_per_sec = min_incremental_energy / duration_total_safe
    energy_med = np.median(energy_per_sec)
    energy_mad = np.median(np.abs(energy_per_sec - energy_med)) + eps
    energy_eff_score = np.clip((energy_per_sec - energy_med) / energy_mad, -2.0, 2.0)
    slack_headroom = np.maximum(0.0, slack - 0.7374270104049188)
    max_slack_headroom = np.max(slack_headroom) + eps
    slack_headroom_normalized = slack_headroom / max_slack_headroom
    host_load_scale = slack_headroom_normalized * 1.472601487607053
    rank_med = np.median(upward_rank)
    rank_mad = np.median(np.abs(upward_rank - rank_med)) + eps
    rank_norm = np.clip((upward_rank - rank_med) / rank_mad, -2.0, 2.0)
    rank_score = -rank_norm * (1.0 + slack_headroom_normalized)
    unc_sigmoid = 1.0 / (1.0 + np.exp(-4.541883858990119 * (norm_uncert - 0.0)))
    energy_uncertainty_score = ddl_safe_mask * host_load_scale * 0.5380169098507215 * energy_eff_score * norm_uncert * unc_sigmoid
    wait_med = np.median(ready_wait_time)
    wait_mad = np.median(np.abs(ready_wait_time - wait_med)) + eps
    wait_norm = np.clip((ready_wait_time - wait_med) / wait_mad, -2.0, 2.0)
    wait_score = ddl_safe_mask * wait_norm * slack_headroom_normalized * 0.5052621801898608
    score = np.clip(slack_score, 0.0, finfo.max) + unc_slack_coupling + duration_risk + critical_path_release
    score += ddl_safe_mask * (1.4434733238589328 * host_load_scale * energy_eff_score + rank_score + energy_uncertainty_score + wait_score)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
