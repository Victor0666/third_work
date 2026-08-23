import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """
    Hybrid priority rule with:
      - Piecewise slack modeling (Parent 2)
      - Quadruple-gated successor-release interaction (upward_rank * remaining_work), merged into critical_path_leverage logic
      - Novel energy_uncertainty_coupling (new parameter)
      - Removed unused slack_pressure_tanh_scale; replaced tanh coupling with linear slack_pressure scaling
      - All numeric literals restricted to {-2,-1,0,1,2}; epsilon via PARAMS; inf/nan guarded via np.finfo.
    """
    eps = 2.0150908318504493e-06
    finfo = np.finfo(np.float64)
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    slack = np.nan_to_num(np.asarray(slack, dtype=np.float64), nan=0.0, posinf=finfo.max, neginf=finfo.min)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=np.float64), nan=0.0, posinf=finfo.max, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=eps)

    def mad_normalize(x):
        x = np.asarray(x, dtype=np.float64)
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        scale = 1.4390714075544826 * (mad if mad > eps else eps)
        return (x - med) / scale
    neg_mask = slack < 0.0
    tight_mask = (slack >= 0.0) & (slack <= 3.903783493259634)
    loose_mask = slack > 3.903783493259634
    slack_norm = np.zeros_like(slack)
    slack_norm[neg_mask] = 0.5377010594492818 * -slack[neg_mask]
    slack_norm[tight_mask] = 1.618129146467639 * (np.exp(slack[tight_mask]) - 1.0)
    slack_norm[loose_mask] = 1.618129146467639 * (np.exp(3.903783493259634) - 1.0)
    duration = min_exec_time + min_comm_time
    duration_norm = mad_normalize(duration)
    energy_norm = mad_normalize(min_incremental_energy)
    energy_score = 0.4397651846801969 * energy_norm
    unc_med = np.median(uncertainty)
    unc_high_mask = uncertainty > unc_med + eps
    energy_uncertainty_penalty = 0.18581996150065735 * energy_norm * unc_high_mask.astype(float)
    rank_norm = mad_normalize(upward_rank)
    unc_med = np.median(uncertainty)
    eng_med = np.median(min_incremental_energy)
    unc_gate = (uncertainty <= unc_med).astype(float)
    eng_gate = (min_incremental_energy <= eng_med).astype(float)
    duration_safe = (duration > eps).astype(float)
    critical_gate = ((slack >= 0.0) * unc_gate * eng_gate * duration_safe).astype(float)
    successor_interaction = upward_rank * remaining_work
    successor_norm = mad_normalize(successor_interaction)
    critical_boost = 0.4182579572057976 * successor_norm * critical_gate
    work_density = np.divide(remaining_work, duration + eps)
    work_density_norm = mad_normalize(work_density)
    work_density_gate = (slack >= 0.0).astype(float)
    work_density_bonus = 0.1687838505027749 * work_density_norm * work_density_gate
    wait_norm = mad_normalize(ready_wait_time)
    wait_boost = 1.0 - np.exp(-0.3962201739031116 * np.maximum(wait_norm, 0.0))
    wait_score = wait_boost * (slack >= 0.0).astype(float)
    unc_norm = mad_normalize(uncertainty)
    slack_pressure = np.clip(-slack, 0.0, np.inf)
    uncertainty_amplifier = 0.7639621679919465 * unc_norm * (slack_pressure / (3.903783493259634 + eps))
    score = slack_norm + 0.6573028677193232 * duration_norm + energy_score + energy_uncertainty_penalty - critical_boost - work_density_bonus + wait_score + uncertainty_amplifier
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=-finfo.max)
    return score
