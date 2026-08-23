import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """
    Self-evolved priority rule featuring:
      - Bounded, per-feature MAD-normalized interactions (e.g., work_density normalized *before* gating)
      - Reinstated tunable slack_pressure_tanh_scale for calibrated uncertainty-pressure coupling
      - Removed fragile upward_rank × remaining_work; replaced with robust, pre-normalized work_density_ratio
      - All interaction terms now use post-normalization gating to prevent noise amplification
      - Strict epsilon-guarded arithmetic and finite-output guarantees.
    """
    eps = 2.48537937760949e-07
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
        scale = 1.5282182061218812 * (mad if mad > eps else eps)
        return (x - med) / scale
    neg_mask = slack < 0.0
    tight_mask = (slack >= 0.0) & (slack <= 46.392894720270334)
    loose_mask = slack > 46.392894720270334
    slack_norm = np.zeros_like(slack)
    slack_norm[neg_mask] = 0.5968548607446954 * -slack[neg_mask]
    slack_norm[tight_mask] = 3.083116696428224 * (np.exp(slack[tight_mask]) - 1.0)
    slack_norm[loose_mask] = 3.083116696428224 * (np.exp(46.392894720270334) - 1.0)
    duration = min_exec_time + min_comm_time
    duration_norm = mad_normalize(duration)
    energy_norm = mad_normalize(min_incremental_energy)
    energy_score = 2.2249094735319233 * energy_norm
    rank_norm = mad_normalize(upward_rank)
    unc_med = np.median(uncertainty)
    eng_med = np.median(min_incremental_energy)
    unc_gate = (uncertainty <= unc_med).astype(float)
    eng_gate = (min_incremental_energy <= eng_med).astype(float)
    critical_gate = ((slack >= 0.0) * unc_gate * eng_gate).astype(float)
    critical_boost = 2.7841406742591706 * rank_norm * critical_gate
    work_density = np.divide(remaining_work, duration + eps)
    work_density_norm = mad_normalize(work_density)
    work_density_gate = (slack >= 0.0).astype(float)
    work_density_bonus = 0.15582124509200185 * work_density_norm * work_density_gate
    wait_norm = mad_normalize(ready_wait_time)
    wait_boost = 1.0 - np.exp(-0.05867045228863178 * np.maximum(wait_norm, 0.0))
    wait_score = wait_boost * (slack >= 0.0).astype(float)
    unc_norm = mad_normalize(uncertainty)
    slack_pressure = np.clip(-slack, 0.0, np.inf)
    uncertainty_amplifier = 2.7335761138502734 * unc_norm * np.tanh(slack_pressure * 0.7429303266999989)
    score = slack_norm + 0.7438593206053419 * duration_norm + energy_score - critical_boost - work_density_bonus + wait_score + uncertainty_amplifier
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=-finfo.max)
    return score
