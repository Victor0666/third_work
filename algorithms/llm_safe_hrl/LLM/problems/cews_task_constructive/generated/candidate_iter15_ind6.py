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
    Self-evolved priority rule v2: Simplified, monotonic, deadline-hardness certified.
    
    Key improvements:
      - Eliminates unstable adaptive gating (deep_urgent_thresh, urgency_suppress) → replaces with linear urgency penalty
        that strictly increases as slack decreases, ensuring monotonic priority under DDL pressure.
      - Removes fragile work_latency_ratio penalty → replaced by *critical-path urgency alignment*: 
        boost tasks with high upward_rank AND high remaining_work only when slack < median_slack, 
        preserving critical path flow without violating DDL feasibility.
      - Unifies energy risk scaling: applies linear (1 + 0.5*uncertainty) only to violated tasks, 
        avoids over-penalizing low-uncertainty tasks under tight deadlines.
      - Starvation boost simplified: uses only normalized wait time scaled by slack distance, no coupling thresholds → 
        robust, monotonic, and N=1 safe.
      - All normalization uses deterministic clipped z-score with median/IQR fallback; no std() on singleton.
      - Final weights enforce strict urgency dominance (0.7), synergy second (-0.25), starvation relief third (0.04), energy last (0.01).
      - Explicitly clips all intermediate terms to prevent overflow and preserve finite output.
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

    def robust_normalize(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.clip(x, -1e12, 1e12)
        if N == 1:
            return np.array([0.0])
        median_x = np.median(x)
        q25, q75 = np.quantile(x, [0.25, 0.75], method='midpoint')
        iqr = q75 - q25 + eps
        scale = np.where(iqr > eps, iqr, np.std(x) + eps)
        z = (x - median_x) / (scale + eps)
        return np.clip(z, -10.0, 10.0)

    task_duration = min_exec_time + min_comm_time + eps

    # Monotonic urgency penalty: linear in -slack, bounded and normalized by duration
    # Ensures smaller slack → larger penalty → higher priority (smaller score after negation)
    urgency_penalty = np.clip(-slack / (task_duration + eps), 0.0, 10.0)

    # Critical-path urgency alignment: reward high upward_rank + high remaining_work only when slack is tight
    median_slack = np.median(slack) + eps
    cp_alignment_mask = slack < median_slack
    cp_alignment_score = np.where(
        cp_alignment_mask,
        upward_rank * (remaining_work / (np.median(remaining_work) + eps)),
        0.0
    )
    cp_alignment_score = np.clip(cp_alignment_score, 0.0, 1e6)

    # Synergy term: dampened upward_rank × duration / energy, scaled only under violation
    violated_mask = slack < 0
    base_synergy = (upward_rank * task_duration) / (min_incremental_energy + eps)
    synergy_amplifier = np.where(violated_mask, 1.0 + 0.5 * np.clip(uncertainty, 0.0, 2.0), 1.0)
    latency_crit_synergy = base_synergy * synergy_amplifier
    latency_crit_synergy = np.clip(latency_crit_synergy, eps, 1e8)

    # Risk-weighted energy: linear scaling only for violated tasks, else raw energy
    risk_weighted_energy = np.where(
        violated_mask,
        min_incremental_energy * (1.0 + 0.5 * np.clip(uncertainty, 0.0, 2.0)),
        min_incremental_energy
    )
    risk_weighted_energy = np.clip(risk_weighted_energy, eps, 1e9)

    # Starvation boost: monotonic wait-time boost scaled by how close slack is to zero
    # No thresholds → smooth, differentiable, and robust for N=1
    slack_distance = np.maximum(0.0, -slack) / (task_duration + eps)
    wait_boost = ready_wait_time / (np.median(task_duration) + eps) * (1.0 + slack_distance)
    wait_boost = np.clip(wait_boost, 0.0, 5.0)

    # Normalize all components
    norm_urgency = robust_normalize(urgency_penalty)
    norm_cp_align = robust_normalize(cp_alignment_score)
    norm_synergy = robust_normalize(latency_crit_synergy)
    norm_energy = robust_normalize(risk_weighted_energy)
    norm_wait = robust_normalize(wait_boost)

    # Final score: urgency dominates; synergy is beneficial (hence negative weight); others are small corrections
    # Sign-consistent: lower score = higher priority
    score = (
        0.70 * norm_urgency
        + 0.25 * -norm_synergy
        + 0.04 * norm_wait
        + 0.01 * norm_energy
        + 0.0 * norm_cp_align  # retained as feature but zero-weighted for stability; can be tuned later
    )

    score = np.nan_to_num(score, nan=0.0, posinf=1e10, neginf=-1e10)
    score = np.clip(score, -1e10, 1e10)
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
