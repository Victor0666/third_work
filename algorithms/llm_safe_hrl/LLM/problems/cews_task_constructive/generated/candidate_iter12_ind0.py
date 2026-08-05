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
    Self-evolved priority rule v2: Deadline-safe dominance + risk-isolated synergy + urgency-adaptive fairness.

    Key improvements over v1:
      - Reduced deadline penalty coefficient (2.5 → 1.8) to suppress noise amplification while preserving hard constraint enforcement.
      - Relaxed starvation gating: now activates for `slack >= -0.1` (sub-second grace) instead of `>= 0`, preventing critical-path delay near DDL.
      - Restored balanced energy-tradeoff: lowered energy_score weight (0.4 → 0.25) and increased synergy_score weight (0.7 → 0.9) to strengthen latency-criticality alignment.
      - Introduced *urgency-aware wait boost*: scales wait_penalty by `max(0, 1 + 5*min(0, slack))` to gently suppress waiting for lateness-risk tasks.
      - Replaced fixed `slack_30` threshold with adaptive `slack_quantile = quantile(slack, 0.2)` for tighter urgency discrimination.
      - Added *work-normalized latency penalty*: penalizes high `remaining_work / duration` only when slack is positive → avoids starving high-work critical paths.
      - All divisions guarded by eps; all NaN/inf replaced via nan_to_num with finite bounds; shape assertion preserved.
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

    def robust_scale(x):
        if len(x) == 0:
            return x
        med = np.median(x)
        x_centered = x - med
        q1, q3 = np.quantile(x_centered, [0.25, 0.75], method='higher')
        iqr = q3 - q1
        if iqr < eps:
            mad = np.median(np.abs(x_centered))
            scale = mad if mad > eps else np.mean(np.abs(x_centered)) + eps
        else:
            scale = iqr + eps
        scaled = x_centered / scale
        return np.clip(scaled, -1000000.0, 1000000.0)

    # Duration: execution + communication, strictly positive
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    duration_med = np.median(duration) + eps
    norm_duration = duration / duration_med

    # Deadline penalty: only for negative slack, uncertainty-weighted, with adaptive urgency boost
    neg_slack = np.maximum(-slack, 0.0)
    uncertainty_factor = 1.0 + 0.7 * np.clip(uncertainty, 0.0, 2.0)
    deadline_penalty = neg_slack * uncertainty_factor

    # Adaptive urgency threshold: 20th percentile (more sensitive than 30th) for early intervention
    slack_sorted = np.sort(slack)
    slack_quantile_idx = max(0, int(0.2 * len(slack_sorted)))
    slack_quantile = slack_sorted[slack_quantile_idx] if N > 0 else 0.0
    is_deeply_urgent = slack < slack_quantile
    deadline_penalty = np.where(is_deeply_urgent, deadline_penalty + 15.0, deadline_penalty)

    # Synergy: critical-path importance × normalized latency ÷ energy → higher = more valuable per joule
    synergy_denom = min_incremental_energy + eps
    synergy = upward_rank * norm_duration / synergy_denom
    synergy_score = -robust_scale(synergy)  # invert so higher synergy → lower score (higher priority)

    # Energy efficiency: smooth relative density penalty only for inefficient tasks
    energy_density = min_incremental_energy / (duration + eps)
    energy_density_med = np.median(energy_density) + eps
    energy_rel_ratio = np.clip(energy_density / energy_density_med, 1.0, None) - 1.0
    energy_base = min_incremental_energy * (1.0 + 0.4 * np.clip(uncertainty, 0.0, 1.0))
    energy_score = robust_scale(energy_base) * energy_rel_ratio

    # Latency penalty: raw duration, robust-scaled → longer duration → higher score (lower priority)
    latency_score = robust_scale(duration)

    # Starvation control: activated for near-deadline & non-top-critical tasks (relaxed slack guard)
    uprank_75 = np.quantile(upward_rank, 0.75) if N > 1 else np.max(upward_rank)
    is_starvable = (slack >= -0.1) & (upward_rank <= uprank_75)  # grace window: -0.1s
    wait_thresh = np.quantile(ready_wait_time, 0.75) if N > 1 else np.max(ready_wait_time)
    wait_std = np.std(ready_wait_time) + eps
    wait_boost_raw = np.where(ready_wait_time > wait_thresh, (ready_wait_time - wait_thresh) / wait_std, 0.0)
    # Urgency-aware scaling: suppress wait boost when slack < 0 (no starvation for urgent tasks)
    urgency_suppress = np.clip(1.0 + 5.0 * np.minimum(0.0, slack), 0.0, 1.0)
    wait_score = -wait_boost_raw * is_starvable.astype(float) * urgency_suppress

    # Work-latency imbalance penalty: penalize low-duration/high-work tasks *only when slack > 0*
    work_latency_ratio = remaining_work / (duration + eps)
    work_latency_ratio_med = np.median(work_latency_ratio) + eps
    work_imbalance = np.where(slack > 0.0, np.clip(work_latency_ratio / work_latency_ratio_med, 1.0, None) - 1.0, 0.0)
    work_score = robust_scale(work_latency_ratio) * work_imbalance

    # Final weighted combination — rebalanced to emphasize synergy & deadline safety
    score = (
        1.8 * deadline_penalty +
        0.9 * synergy_score +
        0.25 * energy_score +
        0.15 * latency_score +
        0.12 * wait_score +
        0.08 * work_score
    )

    # Finite safeguard
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    score = np.clip(score, -1e12, 1e12)
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
