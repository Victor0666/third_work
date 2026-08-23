import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Novel priority rule emphasizing deadline risk mitigation, critical-path leverage,
    and starvation-aware efficiency. Uses piecewise slack handling, robust normalization,
    and dimensionless interaction terms — all deterministic and epsilon-guarded."""
    eps = 0.008299512118850638
    finfo = np.finfo(np.float64)
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    slack = np.nan_to_num(np.asarray(slack, dtype=np.float64), nan=0.0, posinf=finfo.max, neginf=finfo.min)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=np.float64), nan=0.0, posinf=finfo.max, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=eps)

    def robust_normalize(x):
        x = np.abs(x)
        center = np.median(x)
        spread = np.median(np.abs(x - center)) + eps
        return (x - center) / spread
    duration_raw = min_exec_time + min_comm_time
    duration_norm = np.sign(duration_raw) * np.abs(duration_raw) ** 0.20366095060614614
    duration_norm = robust_normalize(duration_norm)
    slack_norm = np.zeros_like(slack)
    negative_mask = slack < 0.0
    positive_mask = slack >= 0.0
    slack_norm[negative_mask] = 9.46242195509467 * slack[negative_mask]
    slack_norm[positive_mask] = -np.power(slack[positive_mask] + eps, 0.8127427690801445)
    energy_norm = -robust_normalize(min_incremental_energy)
    abs_slack = np.abs(slack)
    max_abs_slack = np.max(abs_slack) + eps
    slack_pressure = np.clip(-slack / max_abs_slack, 0.0, 1.0)
    rank_norm = robust_normalize(upward_rank)
    critical_leverage = 1.4794067587770345 * slack_pressure * rank_norm
    work_density = np.divide(remaining_work, duration_raw + eps)
    work_density_norm = robust_normalize(work_density)
    work_density_bonus = 0.4852355859992028 * work_density_norm
    wait_relief = ready_wait_time / (uncertainty + eps)
    wait_norm = robust_normalize(wait_relief)
    starvation_relief = 0.17927649283524133 * wait_norm
    max_uncertainty = np.max(uncertainty) + eps
    uncertainty_mod = np.clip(uncertainty / max_uncertainty, 0.0, 1.0)
    slack_uncertainty_damp = uncertainty_mod * np.where(positive_mask, -slack_norm, 0.0)
    uncertainty_effect = -0.6370334951852336 * slack_uncertainty_damp
    score = slack_norm + energy_norm * 0.6030383395585727 + duration_norm + critical_leverage + work_density_bonus + starvation_relief + uncertainty_effect
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
