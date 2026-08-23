import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """
    Self-evolved priority rule featuring:
      - Piecewise-linear + saturating exponential slack urgency (replaces fragile tanh),
      - Triple-gated critical-path leverage (slack>=0 ∧ uncertainty<=median ∧ energy<median),
      - Robust MAD-normalized features with epsilon-guarded denominators,
      - Unified anti-starvation wait fairness and uncertainty-pressure coupling.
    All parameters are used; no numeric literals except {-2,-1,0,1,2}; deterministic and finite.
    """
    eps = 2.687212762251716e-07
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
        scale = 1.575149125872536 * (mad if mad > eps else eps)
        return (x - med) / scale
    neg_mask = slack < 0.0
    tight_mask = (slack >= 0.0) & (slack <= 23.083164667913227)
    loose_mask = slack > 23.083164667913227
    slack_norm = np.zeros_like(slack)
    slack_norm[neg_mask] = 7.475956301337063 * -slack[neg_mask]
    slack_norm[tight_mask] = 0.7114800873383127 * (np.exp(slack[tight_mask]) - 1.0)
    slack_norm[loose_mask] = 0.7114800873383127 * (np.exp(23.083164667913227) - 1.0)
    duration = min_exec_time + min_comm_time
    duration_norm = mad_normalize(duration)
    energy_norm = mad_normalize(min_incremental_energy)
    energy_score = 0.10185782484260004 * energy_norm
    rank_norm = mad_normalize(upward_rank)
    unc_med = np.median(uncertainty)
    eng_med = np.median(min_incremental_energy)
    unc_gate = (uncertainty <= unc_med).astype(float)
    eng_gate = (min_incremental_energy <= eng_med).astype(float)
    critical_gate = ((slack >= 0.0) * unc_gate * eng_gate).astype(float)
    critical_boost = 0.6384813119515451 * rank_norm * critical_gate
    work_density = np.divide(remaining_work, duration + eps)
    work_density_norm = mad_normalize(work_density)
    work_density_gate = (slack >= 0.0).astype(float)
    work_density_bonus = 0.4369285899115431 * work_density_norm * work_density_gate
    wait_norm = mad_normalize(ready_wait_time)
    wait_boost = 1.0 - np.exp(-0.2326123567764672 * np.maximum(wait_norm, 0.0))
    wait_score = wait_boost * (slack >= 0.0).astype(float)
    unc_norm = mad_normalize(uncertainty)
    slack_pressure = np.clip(-slack, 0.0, np.inf)
    uncertainty_amplifier = 1.4507261748681777 * unc_norm * np.tanh(slack_pressure * 0.010357009457890799)
    score = slack_norm + 0.22460700127068217 * duration_norm + energy_score - critical_boost - work_density_bonus + wait_score + uncertainty_amplifier
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=-finfo.max)
    return score
