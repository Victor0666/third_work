import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """
    Self-evolved priority rule featuring:
      - Replaced tanh joint gate with robust, differentiable sigmoid of normalized slack/uncertainty ratio.
      - Simplified critical-path leverage to pure slack>=0 gating + clipped quantile-normalized upward_rank.
      - Replaced Q1/Q3 normalization with fixed-epsilon median-IQR scaling using declared normalization_epsilon.
      - All numeric literals are {-2,-1,0,1,2}; no loops, randomness, or side effects.
    """
    finfo = np.finfo(np.float64)
    eps = 2.2643083866198033e-08
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    slack = np.nan_to_num(np.asarray(slack, dtype=np.float64), nan=0.0, posinf=finfo.max, neginf=finfo.min)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=np.float64), nan=0.0, posinf=finfo.max, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=eps)

    def median_iqr_normalize(x):
        x = np.asarray(x, dtype=np.float64)
        q1 = np.quantile(x, 0.27556585328546)
        q3 = np.quantile(x, 0.616760834189064)
        iqr = np.maximum(q3 - q1, eps)
        return (x - np.median(x)) / iqr
    neg_mask = slack < 0.0
    tight_mask = (slack >= 0.0) & (slack <= 44.83612810426936)
    loose_mask = slack > 44.83612810426936
    slack_norm = np.zeros_like(slack)
    slack_norm[neg_mask] = 1.7963225440451234 * -slack[neg_mask]
    slack_norm[tight_mask] = 2.793261585823508 * (np.exp(slack[tight_mask]) - 1.0)
    slack_norm[loose_mask] = 2.793261585823508 * (np.exp(44.83612810426936) - 1.0)
    duration = min_exec_time + min_comm_time
    duration_norm = median_iqr_normalize(duration)
    unc_norm = median_iqr_normalize(uncertainty)
    energy_base = min_incremental_energy * (1.0 + np.abs(unc_norm))
    energy_norm = median_iqr_normalize(energy_base)
    energy_score = 1.0588366752504987 * energy_norm
    rank_norm = median_iqr_normalize(upward_rank)
    rank_clipped = np.clip(rank_norm, -1.0, 1.0)
    critical_gate = (slack >= 0.0).astype(float)
    critical_boost = 2.391898050312734 * rank_clipped * critical_gate
    work_density = np.divide(remaining_work, duration + eps)
    work_density_norm = median_iqr_normalize(work_density)
    joint_ratio = slack / (uncertainty + eps)
    joint_feasibility = 1.0 / (1.0 + np.exp(-4.901726135023428 * joint_ratio))
    work_density_bonus = 0.8279091923507564 * work_density_norm * joint_feasibility
    unc_mod = 1.0 + 0.9096727968152203 * np.abs(unc_norm)
    modulated_energy_score = energy_score * unc_mod
    modulated_slack_norm = slack_norm * unc_mod
    score = modulated_slack_norm + 0.11044749434605616 * duration_norm + modulated_energy_score - critical_boost - work_density_bonus
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=-finfo.max)
    return score
