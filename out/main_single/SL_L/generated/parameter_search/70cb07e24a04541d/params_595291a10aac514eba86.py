import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining Parent 2's robustness with Parent 1's anti-starvation expressiveness:
      - Retains Parent 2's bounded duration risk, slack-sigmoid gating, and direct slack-rank coupling.
      - Integrates Parent 1's power-law wait-time boost but gated by slack headroom (not binary), enhancing fairness without compromising DDL safety.
      - Uses joint MAD normalization across all risk dimensions (duration, |slack|, uncertainty) for coherent scaling.
      - All operations guarded against NaN/inf; no in-place mutation; deterministic finite output.
      - Final score structure: [DDL violation] + [tight-slack risk] + [critical path release] + [starvation mitigation (power-law + headroom)] + [energy-uncertainty interaction (smoothly gated)].
    """
    eps = 1.181903642990883e-05
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
    duration_norm = duration_total / (mad_per_dim[0] * 0.12281468327382868 + eps)
    slack_norm = abs_slack / (mad_per_dim[1] * 0.12281468327382868 + eps)
    unc_norm = uncertainty / (mad_per_dim[2] * 0.12281468327382868 + eps)
    slack_penalty = np.where(slack < 0, -slack * (1.0 + 0.8500657448768032 * unc_norm), 0.0)
    duration_risk_base = (min_exec_time + min_comm_time) * uncertainty
    duration_risk_penalty = np.where(slack <= 1.7407472690363535, duration_risk_base * 0.1962490838239589, 0.0)
    critical_release = upward_rank * remaining_work
    critical_release_median = np.median(critical_release)
    critical_release_mad = np.mean(np.abs(critical_release - critical_release_median)) + eps
    critical_release_norm = (critical_release - critical_release_median) / critical_release_mad
    critical_release_score = -critical_release_norm * 1.3981486041309612
    slack_gate = 1.0 / (1.0 + np.exp(-1.1380444751454677 * (slack - 1.7407472690363535)))
    energy_median = np.median(min_incremental_energy)
    energy_mad = np.mean(np.abs(min_incremental_energy - energy_median)) + eps
    energy_norm = (min_incremental_energy - energy_median) / (energy_mad + eps)
    energy_score = slack_gate * energy_norm * 0.38877417862156716
    energy_uncertainty_score = slack_gate * energy_norm * unc_norm * 1.0319906850107194
    slack_headroom = np.clip(slack - 1.7407472690363535, 0.0, np.inf)
    wait_power = ready_wait_time ** 1.213268272854659
    wait_score = wait_power * (1.0 + slack_headroom / (1.7407472690363535 + eps))
    wait_median = np.median(wait_score)
    wait_mad = np.mean(np.abs(wait_score - wait_median)) + eps
    wait_norm = (wait_score - wait_median) / wait_mad
    wait_final = wait_norm * 0.004672334263984561
    slack_distance = np.clip(1.7407472690363535 - slack, 0.0, np.inf)
    slack_rank_score = -upward_rank * (1.0 + slack_distance / (1.7407472690363535 + eps)) * 2.007759496385453
    score = slack_penalty + duration_risk_penalty + critical_release_score + wait_final + slack_rank_score
    score += energy_score + energy_uncertainty_score
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
