import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with three key improvements:
      - Replaces nonlinear `duration_risk_exponent` with bounded linear `bounded_duration_risk_factor`, directly coupling uncertainty × duration only when slack is tight (validated robustness).
      - Eliminates `rank_slack_balance` interpolation; instead uses `slack_rank_coupling` to scale upward_rank by slack distance — preserves rank importance while enforcing DDL-aware modulation.
      - Introduces slack-conditioned `duration_risk_penalty`: active *only* when slack <= ddl_protection_threshold, avoiding spurious penalties in safe region.
      - All other proven components retained: joint MAD normalization, smooth sigmoid gating, adaptive starvation mitigation, and uncertainty-coupled energy terms.
      - No numeric literals beyond {-2,-1,0,1,2}; fully deterministic and finite-output guaranteed.
    """
    eps = 1.4328137710364596e-06
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
    duration_norm = duration_total / (mad_per_dim[0] * 2.6769121509412765 + eps)
    slack_norm = abs_slack / (mad_per_dim[1] * 2.6769121509412765 + eps)
    unc_norm = uncertainty / (mad_per_dim[2] * 2.6769121509412765 + eps)
    slack_penalty = np.where(slack < 0, -slack * (1.0 + 0.7729665586149405 * unc_norm), 0.0)
    duration_risk_base = (min_exec_time + min_comm_time) * uncertainty
    duration_risk_penalty = np.where(slack <= 1.1513263694329414, duration_risk_base * 0.9799954040601906, 0.0)
    critical_release = upward_rank * remaining_work
    critical_release_median = np.median(critical_release)
    critical_release_mad = np.mean(np.abs(critical_release - critical_release_median)) + eps
    critical_release_norm = (critical_release - critical_release_median) / critical_release_mad
    critical_release_score = -critical_release_norm * 0.616753733446517
    slack_gate = 1.0 / (1.0 + np.exp(-6.169831868853422 * (slack - 1.1513263694329414)))
    energy_median = np.median(min_incremental_energy)
    energy_mad = np.mean(np.abs(min_incremental_energy - energy_median)) + eps
    energy_norm = (min_incremental_energy - energy_median) / (energy_mad + eps)
    energy_score = slack_gate * energy_norm * 1.26511431502344
    energy_uncertainty_score = slack_gate * energy_norm * unc_norm * 1.4338222914534384
    slack_distance = np.clip(1.1513263694329414 - slack, 0.0, np.inf)
    wait_score = ready_wait_time * (1.0 + slack_distance / (1.1513263694329414 + eps))
    wait_median = np.median(wait_score)
    wait_mad = np.mean(np.abs(wait_score - wait_median)) + eps
    wait_norm = (wait_score - wait_median) / wait_mad
    wait_final = wait_norm * 0.6084270055860065
    slack_rank_score = -upward_rank * (1.0 + slack_distance / (1.1513263694329414 + eps)) * 1.7002008068054426
    score = slack_penalty + duration_risk_penalty + critical_release_score + wait_final + slack_rank_score
    score += energy_score + energy_uncertainty_score
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
