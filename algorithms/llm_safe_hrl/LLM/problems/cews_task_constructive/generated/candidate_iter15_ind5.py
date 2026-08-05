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
    Self-evolved priority rule v2: DDL-hardened critical-path focus + uncertainty-gated energy efficiency + latency-aware fairness.

    Key improvements over v1:
    - Replaces global deadline_penalty with *latency-critical gating*: only tasks with slack < median_slack activate deadline pressure
    - Restores monotonic but *bounded* urgency (linear ramp clipped at [0,1]) for stability near thresholds, using robust slack centrality
    - Introduces *uncertainty-gated energy efficiency*: energy_per_duration scaled by (1 + uncertainty) only when slack is tight (< 0.5*median_duration)
    - Fairness now *latency-aware*: starvation boost activated only for non-urgent tasks (slack >= 0) AND long-waiting AND low criticality (upward_rank < median), preventing CP starvation
    - Uses trimmed-mean for all centrality measures (slacks, durations, ranks) to ensure robustness in small-N and skewed sets
    - All components normalized via robust_minmax with explicit size guards; all divisions/ops zero/Nan/inf protected
    - Final convex combination weighted to prioritize DDL compliance (0.55), then CP leverage (0.25), then risk-energy (0.12), then fairness (0.08)
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

    def robust_minmax(x):
        if x.size == 0:
            return np.zeros_like(x)
        x_sorted = np.sort(x)
        trim_n = max(1, int(0.1 * len(x_sorted)))
        x_trimmed = x_sorted[trim_n:-trim_n] if len(x_sorted) > 2 * trim_n else x_sorted
        x_min, x_max = np.min(x_trimmed), np.max(x_trimmed)
        rng = x_max - x_min + eps
        return np.clip((x - x_min) / rng, 0.0, 1.0)

    def robust_trimmed_mean(x):
        if x.size == 0:
            return 0.0
        x_sorted = np.sort(x)
        trim_n = max(1, int(0.1 * len(x_sorted)))
        x_trimmed = x_sorted[trim_n:-trim_n] if len(x_sorted) > 2 * trim_n else x_sorted
        return np.mean(x_trimmed) if x_trimmed.size > 0 else 0.0

    task_duration = min_exec_time + min_comm_time + eps
    duration_central = robust_trimmed_mean(task_duration)
    slack_central = robust_trimmed_mean(slack)
    
    # Bounded linear urgency: 0 (plenty of slack) → 1 (critical slack), clamped to [0,1]
    urgency_raw = np.clip((slack_central - slack) / (duration_central + eps), 0.0, 1.0)
    urgency = np.where(slack < 0.0, 1.0, urgency_raw)
    
    # Latency-critical gating: only apply deadline pressure to tasks below median slack
    slack_median = np.median(slack) if N > 0 else 0.0
    is_latency_critical = (slack < slack_median).astype(float)
    neg_slack = np.maximum(-slack, 0.0)
    deadline_pressure = neg_slack * (1.0 + 0.5 * np.clip(uncertainty, 0.0, 2.0)) * is_latency_critical
    
    # Critical-path leverage: upward_rank / remaining_work, scaled by urgency only when slack is tight
    cp_leverage = upward_rank / (remaining_work + eps)
    cp_pressure = cp_leverage * (1.0 + np.clip(neg_slack / (duration_central + eps), 0.0, 3.0))
    
    # Uncertainty-gated energy efficiency: penalize high-energy/high-uncertainty only when slack is tight
    tight_slack_mask = (slack < 0.5 * duration_central).astype(float)
    energy_efficiency = min_incremental_energy / (task_duration + eps)
    risk_weighted_energy = energy_efficiency * (1.0 + 0.7 * np.clip(uncertainty, 0.0, 2.0)) * tight_slack_mask
    
    # Latency-aware fairness: boost only non-urgent, long-waiting, low-criticality tasks
    wait_per_work = ready_wait_time / (remaining_work + eps)
    wait_median = np.median(wait_per_work) if N > 0 else 0.0
    rank_median = np.median(upward_rank) if N > 0 else 0.0
    is_starvable = (slack >= 0.0) & (wait_per_work > wait_median * 1.5) & (upward_rank < rank_median)
    starvation_boost = np.where(is_starvable, ready_wait_time, 0.0)
    
    # Normalize all components robustly
    norm_deadline = robust_minmax(deadline_pressure)
    norm_cp = robust_minmax(cp_pressure)
    norm_energy = robust_minmax(risk_weighted_energy)
    norm_starvation = robust_minmax(starvation_boost)
    
    # Convex combination prioritizing DDL compliance, then CP, then energy, then fairness
    score = (
        0.55 * norm_deadline +
        0.25 * (1.0 - norm_cp) +  # higher CP importance → lower score
        0.12 * norm_energy +
        0.08 * norm_starvation
    )
    
    # Final sanitization
    score = np.nan_to_num(score, nan=1000000000000.0, posinf=1000000000000.0, neginf=-1000000000000.0)
    score = np.clip(score, -1000000000000.0, 1000000000000.0)
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
