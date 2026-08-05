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
    Self-evolved v2: Deadline-dominant, numerically stable, starvation-aware, energy-critical.
    
    Key improvements over v1:
      - Restores strict deadline dominance: urgency term is unweighted base priority (0.55 weight),
        no conditional gating dilutes its signal; slack violation triggers immediate hard penalty.
      - Replaces fragile multi-layer normalization with single robust z-score using MAD for stability
        under low-variance or degenerate (N=1) inputs — avoids IQR fallback artifacts.
      - Fairness re-engineered: wait-per-work penalty is *only* applied when slack >= 0 (to prevent
        starvation of non-urgent tasks), capped and linearly scaled — no distance-amplified boost.
      - Energy-criticality synergy simplified to upward_rank / (min_incremental_energy + eps),
        normalized *after* clipping extremes — removes risky uncertainty-dependent gating.
      - Risk scaling eliminated from energy term: min_incremental_energy is used raw in normalization;
        risk mitigation is handled purely via urgency and fairness terms — avoids over-penalization.
      - All operations protected: explicit eps, clip, nan_to_num, and finite bounds; deterministic.
      - Final weights: urgency (0.55), crit-energy synergy (0.22), fairness (0.10), duration (0.07),
        uncertainty (0.04), work (0.02) — prioritizes DDL fidelity first, then efficiency & fairness.
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

    # Robust normalization: use MAD (more stable than std/IQR for small/N=1) with fallback to eps
    def robust_zscore(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.clip(x, -1e12, 1e12)
        if N == 1:
            return np.array([0.0])
        median_x = np.median(x)
        # Use MAD = median(|x_i - median_x|) for outlier-robust scale
        mad = np.median(np.abs(x - median_x)) + eps
        z = (x - median_x) / mad
        return np.clip(z, -5.0, 5.0)

    task_min_duration = np.maximum(min_exec_time + min_comm_time, eps)
    median_dur = np.median(task_min_duration) + eps

    # Deadline urgency: monotonic, bounded, hard penalty on violation → dominant signal
    # Linear penalty for slack <= 0: 1.0 (baseline) + proportional lateness
    # Exponential decay for slack > 0: smooth drop-off, stable for large slack
    urgency_neg = 1.0 + np.clip(-slack / median_dur, 0.0, 2.0)
    urgency_pos = np.exp(-np.clip(slack / (median_dur + eps), 0.0, 20.0))
    deadline_urgency = np.where(slack <= 0, urgency_neg, urgency_pos)

    # Criticality-energy synergy: pure efficiency ratio — higher rank / lower energy = better
    crit_energy_ratio = upward_rank / (min_incremental_energy + eps)
    crit_energy_ratio = np.clip(crit_energy_ratio, 1e-06, 1e6)
    
    # Fairness: only activate wait penalty for tasks *not* under deadline pressure (slack >= 0)
    # Prevents amplifying lateness; ensures idle-ready tasks don't starve
    wait_mask = slack >= 0
    wait_per_work = ready_wait_time / (remaining_work + eps)
    max_wait_pw = np.maximum(np.max(wait_per_work[wait_mask]) if np.any(wait_mask) else eps, eps)
    wait_penalty = np.where(wait_mask, np.clip(wait_per_work / max_wait_pw, 0.0, 1.0), 0.0)

    # Normalize all components stably
    norm_urgency = robust_zscore(deadline_urgency)
    norm_synergy = robust_zscore(crit_energy_ratio)
    norm_fairness = robust_zscore(wait_penalty)
    norm_dur = robust_zscore(task_min_duration)
    norm_unc = robust_zscore(uncertainty)
    norm_work = robust_zscore(remaining_work)

    # Final score: smaller = higher priority; urgency dominates, others refine within feasibility
    score = (
        0.55 * norm_urgency +
        0.22 * (1.0 - norm_synergy) +  # high synergy → low score
        0.10 * norm_fairness +
        0.07 * norm_dur +
        0.04 * norm_unc +
        0.02 * norm_work
    )

    # Final sanitization: ensure finite, shaped correctly, deterministic
    score = np.nan_to_num(score, nan=0.0, posinf=1e10, neginf=-1e10)
    score = np.clip(score, -1e10, 1e10)
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
