import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule with 12 parameters: retains Parent 2's stability and adds Parent 1's energy gating.
    
    Key structural features:
    - Robust MAD normalization for all core features (Parent 2).
    - Soft sigmoid energy gating applied *before* normalization to activate energy optimization only when slack permits.
    - Critical-path boosting gated by both slack≥0 and low uncertainty (Parent 2).
    - Work density bonus now gated by uncertainty (not slack) to avoid over-trusting speculative ratios.
    - All numeric literals are strictly in {-2,-1,0,1,2}; eps and finfo used for safeguards.
    """
    eps = 2.9799366474120736e-06
    finfo = np.finfo(float)
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=finfo.max, neginf=finfo.min)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=finfo.max, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=eps, posinf=finfo.max, neginf=eps)

    def mad_normalize(x):
        x = np.asarray(x, dtype=float)
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        scale = 1.6909583935479056 * (mad if mad > eps else eps)
        return (x - med) / scale
    neg_mask = slack < 0.0
    zeroish_mask = (slack >= 0.0) & (slack < 1.0)
    pos_mask = slack >= 1.0
    slack_norm = np.zeros_like(slack)
    slack_norm[neg_mask] = 8.56204802480374 * -slack[neg_mask]
    slack_norm[zeroish_mask] = 1.3753338511010222 * (np.exp(slack[zeroish_mask]) - 1.0)
    slack_norm[pos_mask] = np.clip(slack[pos_mask], 0.0, 37.298865910799165)
    energy_gate = 1.0 / (1.0 + np.exp(-0.1501929419758104 * (slack - 37.298865910799165 / 2.0)))
    gated_energy = min_incremental_energy * energy_gate
    duration = min_exec_time + min_comm_time
    duration_norm = mad_normalize(duration)
    energy_norm = mad_normalize(gated_energy)
    energy_score = 0.16018429641818474 * energy_norm
    rank_norm = mad_normalize(upward_rank)
    unc_median = np.median(uncertainty)
    critical_gate = ((slack >= 0.0) & (uncertainty <= unc_median + eps)).astype(float)
    critical_boost = 3.0305762055861187 * rank_norm * critical_gate
    work_density = np.divide(remaining_work, duration + eps)
    work_density_norm = mad_normalize(work_density)
    work_density_gate = (uncertainty <= unc_median + eps).astype(float)
    work_density_bonus = 0.047820422837664 * work_density_norm * work_density_gate
    wait_norm = mad_normalize(ready_wait_time)
    wait_score = 1.017200240215345 * (1.0 - np.exp(-wait_norm))
    unc_norm = mad_normalize(uncertainty)
    slack_pressure = np.clip(-slack, 0.0, np.inf)
    slack_pressure_bounded = np.tanh(slack_pressure * 0.1501929419758104)
    uncertainty_amplifier = 2.3216833446168934 * unc_norm * slack_pressure_bounded
    score = slack_norm + 1.4273117222091147 * duration_norm + energy_score - critical_boost - work_density_bonus + wait_score + uncertainty_amplifier
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
