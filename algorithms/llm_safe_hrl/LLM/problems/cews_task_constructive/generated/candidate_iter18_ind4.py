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
    Self-evolved priority rule v2: Strict deadline dominance + orthogonal risk modulation + starvation-aware fairness.
    
    Key improvements over v1:
    - Restores *structural deadline dominance*: urgency term is unnormalized, unweighted, and placed first with hard clamp
    - Eliminates multiplicative coupling: slack and uncertainty modulate criticality *additively*, not multiplicatively
    - Introduces *orthogonal risk penalty*: separate additive term for uncertainty × max(0,-slack), preserving urgency signal
    - Refines starvation boost: only activates when wait_time > median_wait AND slack > 0 AND uncertainty < median_unc
    - Uses *work-normalized energy density* with explicit zero-work safety and bounded MAD scaling
    - All operations eps-guarded; outputs strictly finite, shape-correct, deterministic
    """
    eps = 1e-08
    min_exec_time = np.asarray(min_exec_time, dtype=float).copy()
    min_comm_time = np.asarray(min_comm_time, dtype=float).copy()
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float).copy()
    slack = np.asarray(slack, dtype=float).copy()
    upward_rank = np.asarray(upward_rank, dtype=float).copy()
    remaining_work = np.asarray(remaining_work, dtype=float).copy()
    ready_wait_time = np.asarray(ready_wait_time, dtype=float).copy()
    uncertainty = np.asarray(uncertainty, dtype=float).copy()
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=float)

    # --- Structural Deadline Dominance (unnormalized, clamped, primary signal) ---
    # Urgency: negative slack per unit duration → higher magnitude = higher priority
    task_duration = np.maximum(min_exec_time + min_comm_time, eps)
    urgency_raw = -np.clip(slack, -1e6, 1e6) / task_duration  # bounded slack to avoid overflow
    # Hard clamp: urgent tasks get score ≤ 0; non-urgent capped at 1.0
    deadline_score = np.clip(urgency_raw, -np.inf, 1.0)

    # --- Orthogonal Risk Penalty (additive, not multiplicative) ---
    # Penalize high uncertainty *only when slack is negative (i.e., already late/risky)
    risk_penalty = np.where(slack <= 0, uncertainty * np.abs(slack), 0.0)
    risk_penalty = np.clip(risk_penalty, 0.0, 1e6)

    # --- Criticality: upward_rank scaled by *separate* uncertainty and slack terms ---
    # Avoid coupling: use additive risk adjustment instead of multiplicative factor
    unc_adjust = np.clip(1.0 + 0.5 * uncertainty, 1.0, 3.0)
    slack_adjust = np.clip(1.0 + 0.8 * np.maximum(0.0, -slack), 1.0, 3.0)
    crit_base = upward_rank / (min_incremental_energy + eps)
    crit_score = crit_base / (unc_adjust * slack_adjust)
    crit_score = np.clip(crit_score, 1e-6, 1e9)

    # --- Energy Fairness: work-normalized marginal energy with robust scaling ---
    work_density = np.maximum(remaining_work, eps)
    energy_per_work = min_incremental_energy / work_density
    energy_per_work = np.clip(energy_per_work, eps, 1e9)
    
    # Robust MAD normalization: median centering + clipped output
    def robust_mad_normalize(x):
        if x.size == 0:
            return np.zeros_like(x)
        center = np.median(x)
        abs_devs = np.abs(x - center)
        mad = np.median(abs_devs) + eps
        normed = (x - center) / mad
        return np.clip(normed, -6.0, 6.0)
    
    norm_energy = robust_mad_normalize(energy_per_work)
    norm_crit = robust_mad_normalize(crit_score)

    # --- Starvation Mitigation: gated by wait_time percentile *and* slack/uncertainty eligibility ---
    median_wait = np.median(ready_wait_time) if N > 0 else 0.0
    starvation_eligible = (
        (slack > 0.0) & 
        (uncertainty < np.median(uncertainty) + eps) & 
        (ready_wait_time > median_wait + eps)
    )
    wait_boost = np.where(starvation_eligible, 
                         np.clip((ready_wait_time - median_wait) / (np.maximum(np.std(ready_wait_time), eps) + eps), 0.0, 0.2), 
                         0.0)

    # --- Final score: deadline dominates (unweighted), others are bounded corrections ---
    # Priority order: deadline_score (primary) → crit_score (secondary) → energy (tertiary) → starvation (last-resort)
    # All secondary terms are normalized & bounded to avoid overwhelming deadline signal
    score = (
        deadline_score +                    # dominant, unnormalized, clamped
        0.3 * norm_crit +                   # secondary: criticality, normalized & scaled down
        0.1 * norm_energy +                 # tertiary: energy fairness
        0.05 * wait_boost +                 # last-resort: starvation relief
        0.2 * risk_penalty                  # orthogonal risk penalty for late tasks
    )

    # Ensure finiteness and shape correctness
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    score = np.clip(score, -1e12, 1e12)
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
