import numpy as np

def get_task_priority_v2(
    min_exec_time,
    min_comm_time,
    min_incremental_energy,
    slack,
    upward_rank,
    remaining_work,
    ready_wait_time,
    uncertainty
):
    """
    Self-evolved v2: Deadline-dominant + critical-path-aligned + starvation-robust.
    
    Key synthesis:
      - Deadline urgency from Parent 2 (hard penalty for slack <= 0, exponential decay for slack > 0)
      - Critical-path density term from Parent 1 (upward_rank / (min_exec_time + min_comm_time + eps)) 
        to prioritize high-impact fast tasks — enhances makespan awareness without coupling to energy risk.
      - Slack-aware energy fairness from Parent 1, but simplified: energy_per_work = min_incremental_energy / (remaining_work + eps),
        then scaled by max(1.0, 1.0 + slack/5.0) to gently reward ample slack and penalize tight slack.
      - Starvation mitigation via wait-per-work *only under positive slack*, capped linearly — same as Parent 2.
      - Unified robust z-score normalization with MAD-based centering and strict clipping; handles N=1 cleanly.
      - Final weights emphasize deadline fidelity (0.48), critical-path efficiency (0.25), energy fairness (0.12),
        critical-path density (0.08), fairness (0.05), and uncertainty (0.02).
    """
    eps = 1e-08
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=float)

    # Robust z-score normalization supporting N=1
    def robust_zscore(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.clip(x, -1e12, 1e12)
        if N == 1:
            return np.array([0.0])
        median_x = np.median(x)
        mad = np.median(np.abs(x - median_x)) + eps
        z = (x - median_x) / mad
        return np.clip(z, -5.0, 5.0)

    # Core task properties
    task_min_duration = np.maximum(min_exec_time + min_comm_time, eps)
    median_dur = np.median(task_min_duration) + eps

    # Deadline urgency: hard penalty for violation (slack <= 0), exponential decay for margin
    urgency_neg = 1.0 + np.clip(-slack / median_dur, 0.0, 3.0)
    urgency_pos = np.exp(-np.clip(slack / (median_dur + eps), 0.0, 20.0))
    deadline_urgency = np.where(slack <= 0, urgency_neg, urgency_pos)

    # Critical-path energy synergy: upward_rank per unit energy (higher = better)
    crit_energy_ratio = upward_rank / (min_incremental_energy + eps)
    crit_energy_ratio = np.clip(crit_energy_ratio, 1e-06, 1e6)

    # Slack-aware energy fairness: penalize high-energy tasks more when slack is tight
    energy_per_work = min_incremental_energy / (remaining_work + eps)
    slack_scale = np.maximum(1.0, 1.0 + slack / 5.0)  # Gentle scaling: +20% at slack=5s, +100% at slack=50s
    fair_energy_score = energy_per_work * slack_scale
    fair_energy_score = np.clip(fair_energy_score, eps, 1e9)

    # Critical-path density: impact per time unit (higher = better priority)
    cp_density = upward_rank / (task_min_duration + eps)
    cp_density = np.clip(cp_density, eps, 1e6)

    # Starvation fairness: wait-per-work only when slack >= 0
    wait_mask = slack >= 0
    wait_per_work = ready_wait_time / (remaining_work + eps)
    max_wait_pw = np.max(wait_per_work[wait_mask]) if np.any(wait_mask) else eps
    wait_penalty = np.where(wait_mask, np.clip(wait_per_work / (max_wait_pw + eps), 0.0, 1.0), 0.0)

    # Normalize all components
    norm_urgency = robust_zscore(deadline_urgency)
    norm_synergy = robust_zscore(crit_energy_ratio)
    norm_fair_energy = robust_zscore(fair_energy_score)
    norm_cp_density = robust_zscore(cp_density)
    norm_fairness = robust_zscore(wait_penalty)
    norm_unc = robust_zscore(uncertainty)

    # Final score: smaller = higher priority
    # Prioritize deadline compliance first, then efficiency, fairness, and density
    score = (
        0.48 * norm_urgency +
        0.25 * (1.0 - norm_synergy) +  # Higher crit_energy_ratio → lower score
        0.12 * norm_fair_energy +
        0.08 * (1.0 - norm_cp_density) +  # Higher cp_density → lower score
        0.05 * norm_fairness +
        0.02 * norm_unc
    )

    # Final safeguard
    score = np.nan_to_num(score, nan=0.0, posinf=1e10, neginf=-1e10)
    score = np.clip(score, -1e10, 1e10)
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
