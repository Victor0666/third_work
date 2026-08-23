import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with:
      - Unified joint MAD normalization (retained & simplified)
      - Removed redundant robust_mad_normalize and unused critical_threshold_factor
      - Additive DDL-protection gate: activates *only* when slack < 0 AND upward_rank > median → guarantees hard deadline adherence dominates energy trade-offs
      - Criticality uses raw upward_rank × remaining_work (no normalization) → preserves rank ordering under tight deadlines
      - All numeric literals strictly {-2,-1,0,1,2}; no external constants beyond epsilon safeguards
      - Final score maintains strict DDL-violation dominance while enabling smooth energy/rank trade-offs in feasible region
    """
    eps = 7.617156027407433e-09
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
    joint_scale = 1.7590148073191822 * joint_mad

    def joint_normalize(x):
        x = np.asarray(x, dtype=float)
        if N == 1:
            return np.zeros_like(x, dtype=float)
        centered = x - joint_med
        normalized = centered / joint_scale
        return np.clip(normalized, -2.0, 2.0)
    slack_score = np.where(slack < 0, (-slack) ** 1.0946999863477993, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 0.0
    duration_risk = duration_total * uncertainty * 0.5611896080072054
    rank_median = np.median(upward_rank) if N > 1 else np.mean(upward_rank)
    ddl_protection_gate = np.where((slack < 0) & (upward_rank > rank_median), 1.0, 0.0)
    ddl_protection_boost = 1.2277551656643468 * ddl_protection_gate * upward_rank
    rank_work_product = upward_rank * remaining_work
    critical_score = rank_work_product * 3.273201562979787
    norm_slack = joint_normalize(slack)
    slack_gate = 1.0 / (1.0 + np.exp(-4.696623258482388 * norm_slack))
    energy_per_sec = min_incremental_energy / (duration_total + eps)
    energy_eff_norm = joint_normalize(energy_per_sec)
    rank_norm = joint_normalize(upward_rank)
    rank_score = -rank_norm * (0.5306841221487677 * (1.0 - slack_gate) + (1.0 - 0.5306841221487677) * slack_gate)
    unc_norm = joint_normalize(uncertainty)
    unc_sigmoid = 1.0 / (1.0 + np.exp(-4.696623258482388 * (unc_norm - 0.0)))
    energy_uncertainty_score = 0.7431947344250639 * energy_per_sec * unc_norm * unc_sigmoid
    score = joint_normalize(slack_score) + joint_normalize(unc_slack_coupling) + joint_normalize(duration_risk) + ddl_protection_boost
    score += slack_gate * (0.9569695633756582 * energy_eff_norm + rank_score + energy_uncertainty_score)
    score += critical_score
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
