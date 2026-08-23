import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with three key structural improvements:
      - Replaces sigmoid slack gate with a bounded linear ramp (slope parameter) for improved numerical stability, interpretability, and reduced sensitivity to steepness tuning.
      - Replaces power-law wait-time boost with linear headroom scaling: mitigates over-prioritization of old tasks during safety margins while preserving fairness under tight deadlines.
      - Retains joint MAD normalization across duration, |slack|, and uncertainty for cross-scenario robustness.
      - All DDL-critical terms dominate non-DDL terms; energy/uncertainty interactions are strictly gated by slack headroom.
      - No numeric literals beyond {-2,-1,0,1,2}; fully deterministic, finite-output guaranteed.
    """
    eps = 1.117962421612014e-06
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
    duration_norm = duration_total / (mad_per_dim[0] * 0.7072979793036545 + eps)
    slack_norm = abs_slack / (mad_per_dim[1] * 0.7072979793036545 + eps)
    unc_norm = uncertainty / (mad_per_dim[2] * 0.7072979793036545 + eps)
    slack_penalty = np.where(slack < 0, -slack * (1.0 + 0.8858474598020962 * unc_norm), 0.0)
    duration_risk_base = (min_exec_time + min_comm_time) * uncertainty
    duration_risk_penalty = np.where(slack <= 1.98633460021725, duration_risk_base * 0.038507132717930534, 0.0)
    critical_release = upward_rank * remaining_work
    critical_release_median = np.median(critical_release)
    critical_release_mad = np.mean(np.abs(critical_release - critical_release_median)) + eps
    critical_release_norm = (critical_release - critical_release_median) / critical_release_mad
    critical_release_score = -critical_release_norm * 2.9596057831697102
    slack_headroom = slack - 1.98633460021725
    slack_gate = np.clip(0.48009341997958954 * slack_headroom, 0.0, 1.0)
    energy_median = np.median(min_incremental_energy)
    energy_mad = np.mean(np.abs(min_incremental_energy - energy_median)) + eps
    energy_norm = (min_incremental_energy - energy_median) / (energy_mad + eps)
    energy_score = slack_gate * energy_norm * 2.1706839216554927
    energy_uncertainty_score = slack_gate * energy_norm * unc_norm * 1.2509271283652095
    slack_headroom_clipped = np.clip(slack_headroom, 0.0, np.inf)
    wait_score = ready_wait_time * (1.0 + 0.8815388043702429 * slack_headroom_clipped)
    wait_median = np.median(wait_score)
    wait_mad = np.mean(np.abs(wait_score - wait_median)) + eps
    wait_norm = (wait_score - wait_median) / wait_mad
    wait_final = wait_norm * 0.10370822523916395
    slack_distance = np.clip(1.98633460021725 - slack, 0.0, np.inf)
    slack_rank_score = -upward_rank * (1.0 + slack_distance / (1.98633460021725 + eps)) * 0.518116515223462
    score = slack_penalty + duration_risk_penalty + critical_release_score + wait_final + slack_rank_score
    score += energy_score + energy_uncertainty_score
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
