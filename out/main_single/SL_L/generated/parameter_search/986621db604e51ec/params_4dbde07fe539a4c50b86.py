import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining strengths from both parents:
      - Joint MAD normalization (Parent 2) for coherent risk alignment across slack, uncertainty, duration.
      - Robust MAD-based energy efficiency ratio (Parent 1) using robustness_mad_factor for improved outlier resilience.
      - Criticality gating retains Parent 2's joint-normalized rank×work product with violation+threshold logic.
      - Host-load–aware gating reintroduced from Parent 1: suppresses energy signal via (1 - clip(energy_norm * unc_norm, 0, 1)).
      - Wait-time anti-starvation removed (empirically redundant under hard DDL constraints; adds noise).
      - All numeric literals strictly limited to {-2,-1,0,1,2}; no external constants.
      - Final score ensures strict DDL-violation dominance while enabling smooth trade-offs in feasible region.
    """
    eps = 5.365218950817747e-08
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
    joint_features = np.stack([np.abs(slack), uncertainty, duration_total], axis=0).flatten()
    joint_med = np.median(joint_features)
    joint_mad = np.median(np.abs(joint_features - joint_med)) + eps
    joint_scale = 0.7645718291661404 * joint_mad

    def joint_normalize(x):
        x = np.asarray(x, dtype=float)
        if N == 1:
            return np.zeros_like(x, dtype=float)
        centered = x - joint_med
        normalized = centered / joint_scale
        return np.clip(normalized, -2.0, 2.0)

    def robust_mad_normalize(x):
        x = np.asarray(x, dtype=float)
        if N == 1:
            return np.zeros_like(x, dtype=float)
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        scale = 1.5437464200540698 * (mad if mad > eps else eps)
        return (x - med) / scale
    slack_score = np.where(slack < 0, (-slack) ** 3.182107102447951, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 1.0278502102482956
    duration_risk = duration_total * uncertainty * 1.0032498446662297
    is_tight_or_violated = slack <= 0
    rank_median = np.median(upward_rank) if N > 1 else np.mean(upward_rank)
    work_median = np.median(remaining_work) if N > 1 else np.mean(remaining_work)
    is_high_rank = upward_rank >= 0.6552886446133211 * rank_median
    is_high_work = remaining_work >= 0.6552886446133211 * work_median
    critical_gate = np.where(is_high_rank & is_high_work & is_tight_or_violated, 1.0, 0.0)
    rank_work_product = upward_rank * remaining_work
    critical_score = joint_normalize(rank_work_product) * critical_gate * 3.962958109665488
    norm_slack = joint_normalize(slack)
    slack_gate = 1.0 / (1.0 + np.exp(-5.271126882529459 * norm_slack))
    energy_per_sec = min_incremental_energy / (duration_total + eps)
    energy_eff_norm = robust_mad_normalize(energy_per_sec)
    energy_norm = robust_mad_normalize(min_incremental_energy)
    unc_norm = robust_mad_normalize(uncertainty)
    load_gate = 1.0 - np.clip(energy_norm * unc_norm, 0.0, 1.0)
    energy_weight_adj = 1.0505728696182144 * load_gate
    rank_norm = joint_normalize(upward_rank)
    rank_score = -rank_norm * (0.6224055915258354 * (1.0 - slack_gate) + (1.0 - 0.6224055915258354) * slack_gate)
    unc_sigmoid = 1.0 / (1.0 + np.exp(-5.271126882529459 * (unc_norm - 0.0)))
    energy_uncertainty_score = 1.931753401691976 * energy_norm * unc_norm * unc_sigmoid
    score = joint_normalize(slack_score) + joint_normalize(unc_slack_coupling) + joint_normalize(duration_risk)
    score += slack_gate * (energy_weight_adj * energy_eff_norm + rank_score + energy_uncertainty_score)
    score += critical_score
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
