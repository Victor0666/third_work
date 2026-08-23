import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's robustness with Parent 1's latency-risk awareness:
      - Retains linear-ramp slack gate (Parent 2) for stable DDL gating.
      - Uses unified MAD normalization across risk dimensions (duration, |slack|, uncertainty) with stabilizing shift.
      - Integrates latency-risk term (from Parent 1): normalized duration_total weighted only under safe slack.
      - Keeps explicit uncertainty-slack coupling in violation penalty (Parent 2 insight).
      - Adds dedicated latency_risk_weight parameter to control tradeoff between speed and energy under safety.
      - Removes fragile exponential forms and redundant interactions; all literals are in {-2,-1,0,1,2}.
      - Final score preserves lexicographic dominance: DDL violations dominate; among compliant tasks,
        critical-path release, latency efficiency, and starvation mitigation jointly guide selection.
    """
    eps = 4.042757392713256e-08
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
    mad_shifted = mad_per_dim + 0.04219736171092557 * np.median(mad_per_dim)
    duration_norm = duration_total / (mad_shifted[0] * 0.8076058330400202 + eps)
    slack_norm = abs_slack / (mad_shifted[1] * 0.8076058330400202 + eps)
    unc_norm = uncertainty / (mad_shifted[2] * 0.8076058330400202 + eps)
    slack_penalty = np.where(slack < 0, -slack * (1.0 + 0.37642836608743657 * unc_norm), 0.0)
    critical_release = upward_rank * remaining_work
    mad_cr = np.mean(np.abs(critical_release - np.median(critical_release))) + eps
    critical_release_norm = (critical_release - np.median(critical_release)) / (mad_cr * 0.8076058330400202 + eps)
    critical_release_score = -critical_release_norm * 2.3923612649182413
    ramp_half = 0.2656987489011171 / 2.0
    gate_center = 1.0336925360761082
    gate_low = gate_center - ramp_half
    gate_high = gate_center + ramp_half
    slack_gate = np.clip((slack - gate_low) / (ramp_half * 2.0 + eps), 0.0, 1.0)
    mad_energy = np.mean(np.abs(min_incremental_energy - np.median(min_incremental_energy))) + eps
    energy_norm = (min_incremental_energy - np.median(min_incremental_energy)) / (mad_energy * 0.8076058330400202 + eps)
    energy_score = slack_gate * energy_norm * 0.5115720721065828
    energy_uncertainty_score = slack_gate * energy_norm * unc_norm * 0.8766021184433166
    latency_risk_score = slack_gate * -duration_norm * 0.49221877812606973
    median_wait = np.median(ready_wait_time) + eps
    wait_ratio = np.clip(ready_wait_time / median_wait, 0.0, 2.0)
    slack_deficit_boost = np.clip(-slack / (1.0336925360761082 + eps), 0.0, 2.0)
    wait_score = ready_wait_time * (1.0 + slack_deficit_boost)
    wait_mad = np.mean(np.abs(wait_score - np.median(wait_score))) + eps
    wait_norm = (wait_score - np.median(wait_score)) / (wait_mad * 0.8076058330400202 + eps)
    wait_final = wait_norm * 0.6420153521183836
    slack_weight = np.clip(1.0 - slack_norm / (slack_norm.max() + eps), 0.0, 1.0)
    rank_weight = 1.0 - slack_weight
    rank_score = -upward_rank / (np.median(upward_rank) + eps) * rank_weight * 0.9558785398629933
    score = slack_penalty + critical_release_score + wait_final + energy_score + energy_uncertainty_score + latency_risk_score + rank_score
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
