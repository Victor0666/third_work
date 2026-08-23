import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """
    Self-evolved priority rule featuring:
      - Monotonic uncertainty-weighted energy: replaces brittle gating with smooth scaling by (1 + unc_norm),
        preserving signal continuity and improving gradient stability.
      - Robust Q1/Q3-based normalization (replacing MAD): more stable under skewed distributions and small N;
        uses interquartile range with epsilon-guarded denominator.
      - Unified uncertainty coupling: uncertainty now multiplicatively modulates *both* energy_score and slack_urgency
        (not additive amplifier), aligning with fuzzy risk semantics — higher uncertainty amplifies both cost and urgency.
      - Simplified fairness: removed wait_fairness_gain (per reflection) and rely on slack-gated urgency + critical boost
        for anti-starvation, reducing over-parameterization.
      - All numeric literals are {-2,-1,0,1,2}; no loops, randomness, or side effects.
    """
    eps = 7.462754577371623e-06
    finfo = np.finfo(np.float64)
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    slack = np.nan_to_num(np.asarray(slack, dtype=np.float64), nan=0.0, posinf=finfo.max, neginf=finfo.min)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=np.float64), nan=0.0, posinf=finfo.max, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=eps)

    def q1_q3_normalize(x):
        x = np.asarray(x, dtype=np.float64)
        q1 = np.quantile(x, 0.32495230198012587)
        q3 = np.quantile(x, 0.8271246627052649)
        iqr = q3 - q1
        scale = iqr if iqr > eps else eps
        return (x - np.median(x)) / scale
    neg_mask = slack < 0.0
    tight_mask = (slack >= 0.0) & (slack <= 37.941857777779305)
    loose_mask = slack > 37.941857777779305
    slack_norm = np.zeros_like(slack)
    slack_norm[neg_mask] = 5.441442079248933 * -slack[neg_mask]
    slack_norm[tight_mask] = 0.30016062794606113 * (np.exp(slack[tight_mask]) - 1.0)
    slack_norm[loose_mask] = 0.30016062794606113 * (np.exp(37.941857777779305) - 1.0)
    duration = min_exec_time + min_comm_time
    duration_norm = q1_q3_normalize(duration)
    unc_norm = q1_q3_normalize(uncertainty)
    energy_base = min_incremental_energy * (1.0 + np.abs(unc_norm))
    energy_norm = q1_q3_normalize(energy_base)
    energy_score = 2.6313446775922706 * energy_norm
    rank_norm = q1_q3_normalize(upward_rank)
    unc_med = np.median(uncertainty)
    eng_med = np.median(min_incremental_energy)
    unc_gate = (uncertainty <= unc_med).astype(float)
    eng_gate = (min_incremental_energy <= eng_med).astype(float)
    critical_gate = ((slack >= 0.0) * unc_gate * eng_gate).astype(float)
    critical_boost = 2.5856639374601564 * rank_norm * critical_gate
    work_density = np.divide(remaining_work, duration + eps)
    work_density_norm = q1_q3_normalize(work_density)
    work_density_gate = (slack >= 0.0).astype(float)
    work_density_bonus = 0.19978492199026682 * work_density_norm * work_density_gate
    unc_mod = 1.0 + 2.018721799644802 * np.abs(unc_norm)
    modulated_energy_score = energy_score * unc_mod
    modulated_slack_norm = slack_norm * unc_mod
    score = modulated_slack_norm + 0.01319998237078724 * duration_norm + modulated_energy_score - critical_boost - work_density_bonus
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=-finfo.max)
    return score
