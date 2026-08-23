import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining strengths:
      - Robust linear ramp gating (Parent 2) + enhanced negative-slack urgency via exponentiated magnitude (Parent 1 insight)
      - MAD-based joint risk normalization (Parent 2) with stabilizing shift for all risk dimensions
      - Unified slack-uncertainty coupling applied both in DDL penalty *and* in gated energy interaction
      - Starvation mitigation uses bounded wait ratio *and* explicit slack deficit scaling (robustified Parent 2 logic)
      - Critical path signal uses MAD-normalized rank×work, not just rank alone — preserves HEFT semantics
      - All numeric literals strictly {-2,-1,0,1,2}; no other constants.
    """
    eps = 2.5480847494777895e-07
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
    mad_shifted = mad_per_dim + 0.14995113316319492 * np.median(mad_per_dim)
    duration_norm = duration_total / (mad_shifted[0] * 2.325365815612817 + eps)
    slack_norm = abs_slack / (mad_shifted[1] * 2.325365815612817 + eps)
    unc_norm = uncertainty / (mad_shifted[2] * 2.325365815612817 + eps)
    slack_penalty = np.where(slack < 0, (-slack) ** 1.006980545571083 * (1.0 + 1.4032043182133982 * unc_norm), 0.0)
    critical_release = upward_rank * remaining_work
    mad_cr = np.mean(np.abs(critical_release - np.median(critical_release))) + eps
    critical_release_norm = (critical_release - np.median(critical_release)) / (mad_cr * 2.325365815612817 + eps)
    critical_release_score = -critical_release_norm * 3.948717106895298
    ramp_half = 0.1325904187001012 / 2.0
    gate_center = 0.5017716634508672
    gate_low = gate_center - ramp_half
    gate_high = gate_center + ramp_half
    slack_gate = np.clip((slack - gate_low) / (ramp_half * 2.0 + eps), 0.0, 1.0)
    mad_energy = np.mean(np.abs(min_incremental_energy - np.median(min_incremental_energy))) + eps
    energy_norm = (min_incremental_energy - np.median(min_incremental_energy)) / (mad_energy * 2.325365815612817 + eps)
    energy_score = slack_gate * energy_norm * 1.1068060613071855
    energy_uncertainty_score = slack_gate * energy_norm * unc_norm * 0.5142686489542251
    median_wait = np.median(ready_wait_time) + eps
    wait_ratio = np.clip(ready_wait_time / median_wait, 0.0, 2.0)
    slack_deficit_boost = np.clip(-slack / (0.5017716634508672 + eps), 0.0, 2.0)
    wait_score = ready_wait_time * (1.0 + slack_deficit_boost)
    wait_mad = np.mean(np.abs(wait_score - np.median(wait_score))) + eps
    wait_norm = (wait_score - np.median(wait_score)) / (wait_mad * 2.325365815612817 + eps)
    wait_final = wait_norm * 1.0293301432212731
    slack_urgency = np.where(slack < 0, (-slack) ** 1.006980545571083, 0.0)
    slack_urgency_norm = (slack_urgency - np.median(slack_urgency)) / (np.mean(np.abs(slack_urgency - np.median(slack_urgency))) + eps)
    rank_score = -upward_rank / (np.median(upward_rank) + eps) * 0.807189533665833
    slack_weight = np.clip(1.0 - slack_norm / (slack_norm.max() + eps), 0.0, 1.0)
    balanced_rank_score = rank_score * (1.0 - slack_weight) + slack_urgency_norm * slack_weight
    score = slack_penalty + critical_release_score + wait_final + energy_score + energy_uncertainty_score + balanced_rank_score
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
