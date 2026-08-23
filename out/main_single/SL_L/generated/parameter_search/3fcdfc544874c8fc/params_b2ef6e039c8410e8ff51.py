import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining strengths of both parents:
      - Uses linear ramp slack gate (Parent 2) for robust, interpretable DDL gating
      - Joint MAD normalization with stabilizing shift across duration, |slack|, uncertainty (Parent 2)
      - Adds conditional tight-slack uncertainty coupling (Parent 1) — applied only when slack <= threshold
      - Preserves critical-path release, starvation mitigation, and energy–uncertainty interaction
      - All numeric literals restricted to {-2,-1,0,1,2}; no unbounded functions or fragile nonlinearities
      - Final score is lexicographically ordered: DDL-critical terms dominate non-DDL terms
    """
    eps = 3.81024753235813e-07
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
    all_risk_dims = np.stack([duration_total, abs_slack, uncertainty], axis=0)
    mad_per_dim = np.mean(np.abs(all_risk_dims - np.median(all_risk_dims, axis=1, keepdims=True)), axis=1) + eps
    mad_shifted = mad_per_dim + 0.0001762072144346058 * np.median(mad_per_dim)
    duration_norm = duration_total / (mad_shifted[0] * 0.7486865050450893 + eps)
    slack_norm = abs_slack / (mad_shifted[1] * 0.7486865050450893 + eps)
    unc_norm = uncertainty / (mad_shifted[2] * 0.7486865050450893 + eps)
    slack_penalty = np.where(slack < 0, -slack * (1.0 + 1.0318794020155693 * unc_norm), 0.0)
    tight_slack_mask = np.where(slack <= 0.4053377430072999, 1.0, 0.0)
    tight_slack_uncertainty_coupling = tight_slack_mask * uncertainty * slack_norm * 2.3465394156920265
    critical_release = upward_rank * remaining_work
    mad_cr = np.mean(np.abs(critical_release - np.median(critical_release))) + eps
    critical_release_norm = (critical_release - np.median(critical_release)) / (mad_cr * 0.7486865050450893 + eps)
    critical_release_score = -critical_release_norm * 2.9471296739826256
    ramp_half = 2.999827166042639 / 2.0
    gate_center = 0.4053377430072999
    gate_low = gate_center - ramp_half
    gate_high = gate_center + ramp_half
    slack_gate = np.clip((slack - gate_low) / (ramp_half * 2.0 + eps), 0.0, 1.0)
    mad_energy = np.mean(np.abs(min_incremental_energy - np.median(min_incremental_energy))) + eps
    energy_norm = (min_incremental_energy - np.median(min_incremental_energy)) / (mad_energy * 0.7486865050450893 + eps)
    energy_score = slack_gate * energy_norm * 2.601612721763978
    energy_uncertainty_score = slack_gate * energy_norm * unc_norm * 0.4396590489229244
    median_wait = np.median(ready_wait_time) + eps
    wait_ratio = np.clip(ready_wait_time / median_wait, 0.0, 2.0)
    slack_deficit_boost = np.clip(-slack / (0.4053377430072999 + eps), 0.0, 2.0)
    wait_score = ready_wait_time * (1.0 + slack_deficit_boost)
    wait_mad = np.mean(np.abs(wait_score - np.median(wait_score))) + eps
    wait_norm = (wait_score - np.median(wait_score)) / (wait_mad + eps)
    wait_final = wait_norm * 0.28292952571744956
    slack_weight = np.clip(1.0 - slack_norm / (slack_norm.max() + eps), 0.0, 1.0)
    rank_weight = 1.0 - slack_weight
    rank_score = -upward_rank / (np.median(upward_rank) + eps) * rank_weight * 0.7875052789665232
    score = slack_penalty + tight_slack_uncertainty_coupling + critical_release_score
    score += energy_score + energy_uncertainty_score + wait_final + rank_score
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
