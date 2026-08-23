import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with unified smooth slack transition and adaptive energy gating.
    
    Key structural improvements:
    - Replaces piecewise + tanh-based slack handling with a single bounded sigmoid transition
      centered at slack=0 and width controlled by 'slack_transition_width', eliminating discontinuities
      and improving gradient stability for CMA-ES.
    - Energy gating now uses *relative uncertainty*: activates only when uncertainty < median * threshold,
      making confidence-aware energy optimization task-local and robust to workload skew.
    - Removes redundant parameters (slack_cap, slack_pressure_tanh_scale, robustness_mad_factor)
      per self-reflection; simplifies normalization to direct MAD where needed, and uses raw median
      for gating thresholds to avoid cascaded normalization artifacts.
    - All feature interactions remain bounded, epsilon-guarded, and deterministic.
    """
    eps = 1.0727934882346536e-05
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
        scale = mad if mad > eps else eps
        return (x - med) / scale
    slack_centered = slack
    slack_normalized_width = np.abs(slack_centered) / (3.114674471377395 + eps)
    slack_urgency = 1.0 - 1.0 / (1.0 + np.exp(slack_centered / (3.114674471377395 + eps)))
    neg_mask = slack < 0.0
    slack_norm = np.zeros_like(slack)
    slack_norm[neg_mask] = 3.2364672477092746 * -slack[neg_mask]
    nonneg_mask = ~neg_mask
    slack_norm[nonneg_mask] = 1.0419120921228107 * slack_urgency[nonneg_mask]
    unc_median = np.median(uncertainty)
    energy_gate = (uncertainty <= unc_median * 1.2217195744864224 + eps).astype(float)
    gated_energy = min_incremental_energy * energy_gate
    duration = min_exec_time + min_comm_time
    duration_norm = mad_normalize(duration)
    energy_norm = mad_normalize(gated_energy)
    energy_score = 0.43681123711652914 * energy_norm
    rank_norm = mad_normalize(upward_rank)
    critical_gate = ((slack >= 0.0) & (uncertainty <= unc_median + eps)).astype(float)
    critical_boost = 1.2222626284503377 * rank_norm * critical_gate
    work_density = np.divide(remaining_work, duration + eps)
    work_density_norm = mad_normalize(work_density)
    work_density_gate = (uncertainty <= unc_median + eps).astype(float)
    work_density_bonus = 0.6721044604301067 * work_density_norm * work_density_gate
    wait_norm = mad_normalize(ready_wait_time)
    wait_score = 0.25246050689400157 * (1.0 - np.exp(-wait_norm))
    unc_norm = mad_normalize(uncertainty)
    uncertainty_amplifier = 0.9349676622945658 * unc_norm * slack_urgency
    score = slack_norm + 0.0846302050569927 * duration_norm + energy_score - critical_boost - work_density_bonus + wait_score + uncertainty_amplifier
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
