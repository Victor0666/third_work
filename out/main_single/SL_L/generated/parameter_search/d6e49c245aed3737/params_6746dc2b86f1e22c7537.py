import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining best elements from both parents:
      - Retains Parent 2's strict lexicographic grounding via *smooth* sigmoid DDL gating (not hard mask) for CMA-ES differentiability & stability.
      - Keeps power-law anti-starvation and empirical slack bounds for robustness to small-N and heterogeneity.
      - Adopts Parent 1's joint MAD normalization for duration/slack/uncertainty coherence, but applied only to risk dimensions.
      - Uses critical_path_release_weight (Parent 1) instead of criticality_boost (Parent 2) for clearer semantics and smoother gradient.
      - Replaces hard `slack > 0` gate with differentiable `sigmoid(slack)` to preserve gradient flow during offline co-evolution.
      - All divisions and norms guarded; no in-place mutation; finite output guaranteed.
      - Final score: DDL-critical terms dominate unconditionally; non-DDL terms are smoothly gated by slack headroom.
    """
    eps = 1.0359333221963129e-09
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
    duration_norm = duration_total / (mad_per_dim[0] + eps)
    slack_norm = abs_slack / (mad_per_dim[1] + eps)
    unc_norm = uncertainty / (mad_per_dim[2] + eps)
    slack_score = np.where(slack < 0, (-slack) ** 1.5257306786669824, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 1.1634631474367878
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 0.147802478193509
    critical_release = upward_rank * remaining_work
    critical_release_median = np.median(critical_release)
    critical_release_mad = np.mean(np.abs(critical_release - critical_release_median)) + eps
    critical_release_norm = (critical_release - critical_release_median) / critical_release_mad
    successor_release_contribution = -critical_release_norm * 1.3608823214872632
    ddl_headroom_gate = 1.0 / (1.0 + np.exp(-1.5257306786669824 * slack))
    duration_total_safe = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total_safe
    energy_eff_median = np.median(energy_per_sec)
    energy_eff_mad = np.mean(np.abs(energy_per_sec - energy_eff_median)) + eps
    energy_eff_norm = (energy_per_sec - energy_eff_median) / (energy_eff_mad + eps)
    energy_eff_score = ddl_headroom_gate * energy_eff_norm * 0.9677944824116552
    slack_lb = -0.19367167868300328
    slack_ub = 48.309937005906505
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    weight_rank = 0.5749437312131832 + (1.0 - 0.5749437312131832) * (1.0 - slack_scaled)
    rank_median = np.median(upward_rank)
    rank_mad = np.mean(np.abs(upward_rank - rank_median)) + eps
    rank_norm = (upward_rank - rank_median) / (rank_mad + eps)
    rank_score = -ddl_headroom_gate * rank_norm * weight_rank
    unc_sigmoid = 1.0 / (1.0 + np.exp(-5.117476320110348 * (uncertainty - 1.0)))
    energy_norm = (min_incremental_energy - np.median(min_incremental_energy)) / (np.mean(np.abs(min_incremental_energy - np.median(min_incremental_energy))) + eps)
    unc_norm_smooth = (uncertainty - np.median(uncertainty)) / (np.mean(np.abs(uncertainty - np.median(uncertainty))) + eps)
    energy_uncertainty_score = ddl_headroom_gate * 1.2031416146126188 * energy_norm * unc_norm_smooth * unc_sigmoid
    wait_power = ready_wait_time ** 1.707315119170794
    wait_median = np.median(wait_power)
    wait_mad = np.mean(np.abs(wait_power - wait_median)) + eps
    wait_norm = (wait_power - wait_median) / (wait_mad + eps)
    wait_score = ddl_headroom_gate * wait_norm
    score = slack_score + unc_slack_coupling + duration_risk + successor_release_contribution
    score += energy_eff_score + rank_score + energy_uncertainty_score + wait_score
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
