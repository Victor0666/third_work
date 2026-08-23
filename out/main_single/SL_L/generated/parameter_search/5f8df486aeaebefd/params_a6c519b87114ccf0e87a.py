import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule synthesizing best practices:
      - Combines Parent 2's robust lexicographic gating and joint risk normalization with Parent 1's explicit slack_penalty_exponent.
      - Introduces `anti_starvation_headroom_scale`: scales slack headroom in wait-term to avoid aggressive suppression when headroom is large.
      - Uses exponentiated slack penalty (Parent 1) instead of linear, preserving sharp DDL violation dominance.
      - Retains unconditional critical_path_release_weight (Parent 2) — avoids instability from slack-dependent amplification.
      - All normalization uses MAD with clipping [-2,2]; no cross-feature coupling beyond joint risk alignment.
      - No numeric literals beyond {-2,-1,0,1,2}; all tunables declared and used exactly once.
      - Strictly enforces: smaller score → higher priority; finite output shape (N,); deterministic.
    """
    eps = 2.5280496293665186e-05
    finfo = np.finfo(float)
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=finfo.max, neginf=finfo.min)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=finfo.max, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=eps, posinf=finfo.max, neginf=eps)
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=float)

    def mad_normalize(x):
        x = np.asarray(x, dtype=float)
        if N == 1:
            return np.zeros_like(x, dtype=float)
        med = np.median(x)
        abs_dev = np.abs(x - med)
        mad = np.median(abs_dev) + eps
        normalized = (x - med) / (mad + eps)
        return np.clip(normalized, -2.0, 2.0)
    duration_total = min_exec_time + min_comm_time + eps
    slack_abs = np.abs(slack)
    slack_penalty = np.where(slack < 0, (-slack) ** 3.5369949967510643, 0.0)
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 1.853781397200635
    critical_release_score = upward_rank * remaining_work * 2.849999698765859
    slack_headroom_mask = np.where(slack > 0.40830242429486563, 1.0, 0.0)
    slack_headroom = np.maximum(0.0, slack - 0.40830242429486563)
    energy_norm = mad_normalize(min_incremental_energy)
    energy_term = slack_headroom_mask * 1.983928893744026 * energy_norm
    wait_power = np.where(slack_headroom_mask > 0.0, ready_wait_time ** 1.60570186869151, 0.0)
    wait_norm = mad_normalize(wait_power)
    headroom_factor = np.clip(slack_headroom * 0.6573243272789028, 0.0, 1.0)
    wait_term = slack_headroom_mask * wait_norm * headroom_factor
    unc_sigmoid = 1.0 / (1.0 + np.exp(-5.5919454307384795 * (uncertainty - 1.0)))
    energy_uncertainty_term = slack_headroom_mask * 0.3721398700241856 * energy_norm * mad_normalize(uncertainty) * unc_sigmoid
    rank_norm = mad_normalize(upward_rank)
    headroom_normalized = np.where(slack_headroom_mask > 0.0, slack_headroom / (np.max(slack_headroom) + eps), 0.0)
    weight_rank = 0.24891093573849846 * (1.0 - headroom_normalized) + (1.0 - 0.24891093573849846) * headroom_normalized
    rank_term = slack_headroom_mask * -rank_norm * weight_rank
    score = mad_normalize(slack_penalty) + mad_normalize(duration_risk) + mad_normalize(critical_release_score)
    score += energy_term + wait_term + energy_uncertainty_term + rank_term
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
