import numpy as np

# 函数名中的 2 由 SeEvo 在生成 Prompt 时替换为目标版本号。
# 八个输入均为长度 N 的一维数组；相同下标始终指向同一个 ready task。
# 返回值也必须是长度 N 的一维数组，并遵守“分数越小，优先级越高”。
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
    Self-evolved priority rule v2: Restores critical-path dominance while preserving adaptive deadline safety.
    
    Key improvements over v1:
      - Restores deadline urgency weight to 0.45 (down from 0.65) to rebalance with critical-path alignment (-0.35).
      - Reverts to *division-based risk weighting* (upward_rank / (1 + max(0,-slack)*uncertainty)) for outlier robustness.
      - Introduces *slack-aware energy fairness*: energy normalized by (remaining_work * max(1, 1 + slack/10)), 
        penalizing high-energy tasks more when slack is tight and rewarding them when slack is ample.
      - Starvation boost now applied *after* normalization and tightly clipped [0, 0.15] to avoid priority inversion.
      - Robust normalization uses median-centering + MAD *only if N>=3*, else simple min-max scaling for stability at small N.
      - Adds *critical-path density* term: upward_rank / (min_exec_time + min_comm_time + eps), prioritizing high-impact fast tasks.
      - All clipping and eps-protection applied before normalization to preserve semantic meaning.
      - Final score bounded to [-1e9, 1e9] with deterministic fallback for edge cases.
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
    
    # Pre-normalization clipping and sanitization
    min_exec_time = np.clip(np.nan_to_num(min_exec_time, nan=eps, posinf=eps, neginf=eps), eps, 1e9)
    min_comm_time = np.clip(np.nan_to_num(min_comm_time, nan=eps, posinf=eps, neginf=eps), eps, 1e9)
    min_incremental_energy = np.clip(np.nan_to_num(min_incremental_energy, nan=eps, posinf=eps, neginf=eps), eps, 1e9)
    slack = np.nan_to_num(slack, nan=0.0, posinf=1e9, neginf=-1e9)
    upward_rank = np.clip(np.nan_to_num(upward_rank, nan=eps, posinf=1e6, neginf=eps), eps, 1e6)
    remaining_work = np.clip(np.nan_to_num(remaining_work, nan=eps, posinf=1e9, neginf=eps), eps, 1e9)
    ready_wait_time = np.clip(np.nan_to_num(ready_wait_time, nan=0.0, posinf=1e6, neginf=0.0), 0.0, 1e6)
    uncertainty = np.clip(np.nan_to_num(uncertainty, nan=0.0, posinf=1e3, neginf=0.0), 0.0, 1e3)
    
    # Adaptive deadline urgency: scale-invariant, tanh-clipped, but reduced weight
    task_duration = min_exec_time + min_comm_time
    norm_slack_pressure = -np.clip(slack, -1e6, 1e6) / (task_duration + eps)
    urgency_cliff = np.tanh(norm_slack_pressure * 0.6)
    deadline_urgency = 1.0 + 0.8 * np.maximum(0.0, urgency_cliff)  # softer gating
    
    # Critical-path alignment: division-based risk weighting (robust to outliers)
    slack_risk_penalty = 1.0 + np.maximum(0.0, -slack) * uncertainty
    crit_alignment = upward_rank / (slack_risk_penalty * (min_incremental_energy + eps))
    crit_alignment = np.clip(crit_alignment, eps, 1e7)
    
    # Slack-aware energy fairness: reward low-energy tasks more when slack > 0
    slack_factor = np.maximum(1.0, 1.0 + slack / 10.0)  # amplifies fairness when slack abundant
    work_slack_scaled = remaining_work * slack_factor
    energy_per_work_scaled = min_incremental_energy / (work_slack_scaled + eps)
    energy_per_work_scaled = np.clip(energy_per_work_scaled, eps, 1e9)
    
    # Critical-path density: prioritize high-rank tasks that execute quickly
    cp_density = upward_rank / (task_duration + eps)
    cp_density = np.clip(cp_density, eps, 1e7)
    
    # Starvation eligibility: only when safe (positive slack & low uncertainty)
    median_unc = np.median(uncertainty) + eps
    wait_eligible = (slack > 0.0) & (uncertainty < median_unc)
    max_wait_safe = np.max(ready_wait_time[wait_eligible]) if np.any(wait_eligible) else eps
    wait_boost_raw = np.where(wait_eligible, ready_wait_time / (max_wait_safe + eps), 0.0)
    wait_penalty = np.clip(wait_boost_raw, 0.0, 0.15)  # tighter bound than v1
    
    # Robust normalization: median-MAD for N>=3, min-max otherwise
    def robust_normalize(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.clip(x, -1e9, 1e9)
        if N == 1:
            return np.array([0.0])
        elif N >= 3:
            center = np.median(x)
            abs_devs = np.abs(x - center)
            mad = np.median(abs_devs) + eps
            z = (x - center) / mad
            return np.clip(z, -6.0, 6.0)
        else:
            x_min, x_max = np.min(x), np.max(x)
            if x_max - x_min < eps:
                return np.zeros_like(x)
            return (x - x_min) / (x_max - x_min + eps) * 2.0 - 1.0
    
    norm_urgency = robust_normalize(deadline_urgency)
    norm_crit = robust_normalize(crit_alignment)
    norm_energy = robust_normalize(energy_per_work_scaled)
    norm_cp_density = robust_normalize(cp_density)
    norm_wait = robust_normalize(wait_penalty)
    norm_uncertainty = robust_normalize(uncertainty)
    
    # Final weighted fusion: deadline safety (0.45), critical-path dominance (-0.35), 
    # energy fairness (0.12), CP density (0.08), starvation relief (0.05), uncertainty (-0.03), others neutral
    score = (
        0.45 * norm_urgency 
        - 0.35 * norm_crit 
        + 0.12 * norm_energy 
        + 0.08 * norm_cp_density 
        + 0.05 * norm_wait 
        - 0.03 * norm_uncertainty
    )
    
    # Final sanitization
    score = np.nan_to_num(score, nan=0.0, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e9, 1e9)
    
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
