import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """
    Hybrid priority rule with 12 parameters:
      - Preserves Parent 2's piecewise-smooth slack modeling
      - Adds successor-release interaction (upward_rank * remaining_work) as gated term
      - Removes redundant 'energy_uncertainty_coupling' and 'slack_pressure_tanh_scale'
        to comply with parameter count limit
      - Keeps triple-gated critical-path leverage, work-density bonus, and unified wait fairness
      - All features MAD-normalized; epsilon-guarded; finite-output guaranteed.
    """
    eps = 1.9590989998704393e-06
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
        scale = 1.6579968014038156 * (mad if mad > eps else eps)
        return (x - med) / scale
    neg_mask = slack < 0.0
    tight_mask = (slack >= 0.0) & (slack <= 19.54751345834092)
    loose_mask = slack > 19.54751345834092
    slack_norm = np.zeros_like(slack)
    slack_norm[neg_mask] = 6.343338598683225 * -slack[neg_mask]
    slack_norm[tight_mask] = 1.2795875471115545 * (np.exp(slack[tight_mask]) - 1.0)
    slack_norm[loose_mask] = 1.2795875471115545 * (np.exp(19.54751345834092) - 1.0)
    duration = min_exec_time + min_comm_time
    duration_norm = mad_normalize(duration)
    energy_norm = mad_normalize(min_incremental_energy)
    energy_score = 1.2454777865589108 * energy_norm
    rank_norm = mad_normalize(upward_rank)
    unc_med = np.median(uncertainty)
    eng_med = np.median(min_incremental_energy)
    unc_gate = (uncertainty <= unc_med).astype(float)
    eng_gate = (min_incremental_energy <= eng_med).astype(float)
    critical_gate = ((slack >= 0.0) * unc_gate * eng_gate).astype(float)
    critical_boost = 1.1057855639139227 * rank_norm * critical_gate
    successor_interaction = upward_rank * remaining_work
    successor_norm = mad_normalize(successor_interaction)
    successor_gate = (slack >= 0.0).astype(float)
    successor_boost = 0.47072332635032776 * successor_norm * successor_gate
    work_density = np.divide(remaining_work, duration + eps)
    work_density_norm = mad_normalize(work_density)
    work_density_gate = (slack >= 0.0).astype(float)
    work_density_bonus = 0.7085523135617593 * work_density_norm * work_density_gate
    wait_norm = mad_normalize(ready_wait_time)
    wait_boost = 1.0 - np.exp(-0.16139842735224874 * np.maximum(wait_norm, 0.0))
    wait_score = wait_boost * (slack >= 0.0).astype(float)
    unc_norm = mad_normalize(uncertainty)
    slack_pressure = np.clip(-slack, 0.0, np.inf)
    uncertainty_amplifier = 2.521141482676561 * unc_norm * np.tanh(slack_pressure * 1.0)
    score = slack_norm + 1.0738075778184861 * duration_norm + energy_score - critical_boost - successor_boost - work_density_bonus + wait_score + uncertainty_amplifier
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=-finfo.max)
    return score
