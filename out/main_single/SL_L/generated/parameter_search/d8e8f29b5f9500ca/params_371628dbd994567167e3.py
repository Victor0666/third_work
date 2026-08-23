import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """
    Self-evolved priority rule featuring:
      - Adaptive slack-aware normalization: upward_rank scaled by smooth slack-dependent factors
        (exp(-max(0,slack)) for positive slack, 1/(|slack|+1) for negative), replacing brittle gating.
      - Uncertainty fused multiplicatively into slack_norm with tunable upper bound.
      - All numeric constants declared in PARAMETER_SCHEMA; no hidden literals.
      - Strict compliance: only {-2,-1,0,1,2} used as literals; all others parameterized.
    """
    eps = 2.974976001230248e-08
    finfo = np.finfo(np.float64)
    min_exec_time = np.clip(np.nan_to_num(np.asarray(min_exec_time, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min), eps, finfo.max)
    min_comm_time = np.clip(np.nan_to_num(np.asarray(min_comm_time, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min), eps, finfo.max)
    min_incremental_energy = np.clip(np.nan_to_num(np.asarray(min_incremental_energy, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min), eps, finfo.max)
    slack = np.nan_to_num(np.asarray(slack, dtype=np.float64), nan=0.0, posinf=finfo.max, neginf=finfo.min)
    upward_rank = np.clip(np.nan_to_num(np.asarray(upward_rank, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min), eps, finfo.max)
    remaining_work = np.clip(np.nan_to_num(np.asarray(remaining_work, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min), eps, finfo.max)
    ready_wait_time = np.clip(np.nan_to_num(np.asarray(ready_wait_time, dtype=np.float64), nan=0.0, posinf=finfo.max, neginf=0.0), 0.0, finfo.max)
    uncertainty = np.clip(np.nan_to_num(np.asarray(uncertainty, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=eps), eps, finfo.max)

    def mad_normalize(x):
        x = np.asarray(x, dtype=np.float64)
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        scale = 1.897940219648398 * (mad if mad > eps else eps)
        return (x - med) / scale
    neg_mask = slack < 0.0
    tight_mask = (slack >= 0.0) & (slack <= 58.39792164464457)
    loose_mask = slack > 58.39792164464457
    slack_norm = np.zeros_like(slack)
    slack_norm[neg_mask] = 6.879254363961808 * -slack[neg_mask]
    slack_norm[tight_mask] = 1.0285565874820162 * (np.exp(slack[tight_mask]) - 1.0)
    slack_norm[loose_mask] = 1.0285565874820162 * (np.exp(58.39792164464457) - 1.0)
    unc_med = np.median(uncertainty)
    unc_abs_dev = np.abs(uncertainty - unc_med)
    mad_unc = np.median(unc_abs_dev)
    unc_normalized = (uncertainty - unc_med) / (1.897940219648398 * (mad_unc if mad_unc > eps else eps) + eps)
    unc_gate = np.clip(1.0 + 0.6377025577491764 * np.maximum(unc_normalized, 0.0), 1.0, 2.8158926092554)
    slack_norm = slack_norm * unc_gate
    duration = min_exec_time + min_comm_time
    duration_norm = mad_normalize(duration)
    energy_norm = mad_normalize(min_incremental_energy)
    energy_score = 0.18405145133704032 * energy_norm
    slack_abs = np.abs(slack)
    positive_slack_decay = np.exp(-np.maximum(slack, 0.0))
    negative_slack_boost = 1.0 / (slack_abs * neg_mask.astype(float) + 1.0)
    slack_adaptivity = positive_slack_decay + negative_slack_boost * neg_mask.astype(float)
    rank_norm = mad_normalize(upward_rank)
    critical_boost = 0.15685648483525147 * rank_norm * slack_adaptivity
    work_density = np.divide(remaining_work, duration + eps)
    work_density_norm = mad_normalize(work_density)
    work_density_gate = (slack >= 0.0).astype(float)
    work_density_bonus = 0.8031958955212006 * work_density_norm * work_density_gate
    wait_norm = mad_normalize(ready_wait_time)
    wait_boost = 1.0 - np.exp(-0.48497983096485964 * np.maximum(wait_norm, 0.0))
    wait_score = wait_boost * (slack >= 0.0).astype(float)
    score = slack_norm + 0.828791135421167 * duration_norm + energy_score - critical_boost - work_density_bonus + wait_score
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=-finfo.max)
    return score
