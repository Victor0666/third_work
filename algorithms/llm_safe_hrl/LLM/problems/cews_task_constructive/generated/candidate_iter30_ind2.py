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
    v2 priority rule: Hard-deadline dominance + adaptive slack-scaled criticality +
                      energy-risk gating with deadline proximity + starvation-resilient fairness +
                      uncertainty-coupling only under dual feasibility + robust normalization +
                      unified lateness dominance + normalized work-consolidation boost +
                      latency-aware energy efficiency + dynamic fairness throttling.

    Key self-evolved improvements:
    - Replaces fixed large-magnitude lateness penalty with *latency-aware urgency ramp*: 
      linear penalty slope increases with |slack| but saturates at -1e12 to prevent overflow.
    - Introduces *dynamic fairness throttling*: wait-time relief is gated by both slack > median 
      AND ready_wait_time > median, preventing premature boosting of idle-but-non-urgent tasks.
    - Enhances *energy efficiency* with dual-mode: (i) aggressive boost (rel_slack > 0.8) for low-energy VMs, 
      (ii) conservative penalty (rel_slack < 0.2) for high-energy VMs near deadline.
    - Refines *uncertainty-work coupling*: uses log(1+uncertainty) / (1 + norm_remaining_work) for better scaling,
      and applies soft gating via slack > 0 AND upward_rank > 0.3*max(upward_rank).
    - Adds *critical-path stability term*: penalizes tasks with high upward_rank but low slack margin 
      using norm_upward_rank * max(0, 0.5 - rel_slack), preventing risky early scheduling.
    - All components use robust_minmax_norm with percentile clipping; all divisions guarded; deterministic.
    """
    eps = 1e-08
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    slack = np.nan_to_num(np.asarray(slack, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=np.float64)

    def robust_minmax_norm(x):
        if x.size == 0:
            return np.zeros_like(x)
        if N == 1:
            return np.zeros_like(x)
        p01 = np.percentile(x, 1.0, method='lower')
        p99 = np.percentile(x, 99.0, method='higher')
        x_clipped = np.clip(x, p01, p99)
        x_min = np.min(x_clipped)
        x_max = np.max(x_clipped)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        return (x_clipped - x_min) / (x_max - x_min + eps)

    # Latency-aware urgency ramp: linear up to saturation, avoids overflow
    lateness_mask = slack <= 0.0
    abs_slack = np.abs(slack)
    # Ramp: -1e11 * (1 + 0.02 * abs_slack), capped at -1e12
    lateness_ramp = -1e11 * (1.0 + 0.02 * abs_slack)
    lateness_penalty = np.where(lateness_mask, np.clip(lateness_ramp, -1e12, 0.0), 0.0)

    duration = min_exec_time + min_comm_time + eps
    critical_timing = duration * upward_rank
    norm_critical_timing = robust_minmax_norm(critical_timing)
    
    rel_slack = np.divide(slack, duration, out=np.zeros_like(slack), where=duration != 0)
    rel_slack = np.nan_to_num(rel_slack, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Slack-scaled critical timing: suppress when rel_slack > 0.95 or < 0
    slack_scale = np.clip(1.0 - np.clip(rel_slack, 0.0, 0.95), 0.0, 1.0)
    scaled_critical_timing = norm_critical_timing * slack_scale

    # Critical-path stability penalty: penalize high-rank tasks with tight slack
    cp_stability_penalty = norm_critical_timing * np.clip(0.5 - rel_slack, 0.0, 0.5)

    energy_density = np.divide(min_incremental_energy, duration, out=np.zeros_like(min_incremental_energy), where=duration != 0)
    energy_density = np.nan_to_num(energy_density, nan=0.0, posinf=0.0, neginf=0.0)
    norm_energy_density = robust_minmax_norm(energy_density)

    # Dual-mode energy gating: aggressive boost when slack ample, penalty when tight
    efficiency_boost_mask = (slack > 0.0) & (rel_slack > 0.8)
    efficiency_boost = -0.35 * norm_energy_density * efficiency_boost_mask
    energy_penalty_mask = (slack > 0.0) & (rel_slack < 0.2)
    energy_penalty = 0.2 * norm_energy_density * energy_penalty_mask

    # Fairness: dynamic throttling — only boost if both slack > median AND wait > median
    slack_median = np.median(slack) if N > 1 else slack[0]
    wait_median = np.median(ready_wait_time) if N > 1 else ready_wait_time[0]
    fairness_mask = (slack > slack_median) & (ready_wait_time > wait_median)
    norm_wait_time = robust_minmax_norm(ready_wait_time)
    norm_rw = robust_minmax_norm(remaining_work)
    # Wait relief scaled by work density: longer waits matter more for smaller sub-DAGs
    wait_relief = norm_wait_time / (np.clip(norm_rw, 0.1, 10.0) + eps)
    wait_relief = np.where(np.isfinite(wait_relief), wait_relief, 0.0)
    fairness_boost = wait_relief * fairness_mask

    # Uncertainty-work coupling: log-normalized for stability, soft gating
    ur_max = np.max(upward_rank) if N > 0 else 1.0
    unc_mask = (slack > 0.0) & (upward_rank > 0.3 * ur_max) & (uncertainty > 0.0)
    log_uncertainty = np.log1p(uncertainty)  # log(1+x) avoids log(0)
    norm_rw_for_unc = np.clip(robust_minmax_norm(remaining_work), 0.01, 10.0)
    unc_work_density = log_uncertainty / (1.0 + norm_rw_for_unc + eps)
    unc_work_density = np.where(np.isfinite(unc_work_density), unc_work_density, 0.0)
    norm_unc_work_density = robust_minmax_norm(unc_work_density)
    unc_work_boost = -0.12 * norm_unc_work_density * unc_mask

    # Aggregate score: weights sum to 1.0
    score = (
        0.38 * scaled_critical_timing +
        0.18 * robust_minmax_norm(remaining_work) +
        0.12 * energy_penalty +
        0.09 * fairness_boost +
        0.08 * cp_stability_penalty +
        0.07 * unc_work_boost +
        0.05 * robust_minmax_norm(uncertainty) +
        efficiency_boost
    )

    # Apply hard-deadline dominance via conditional assignment
    score = np.where(lateness_mask, lateness_penalty, score)

    # Final sanitization
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    score = np.clip(score, -1e12, 1e12)

    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
