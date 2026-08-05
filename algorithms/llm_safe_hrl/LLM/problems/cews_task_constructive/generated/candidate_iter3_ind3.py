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

    'Self-evolved priority rule: deadline-smoothed urgency, energy-per-second efficiency with risk-aware clipping,\n     critical-path leverage via normalized slack-adjusted rank, and starvation guard with dynamic activation.\n\n     Key evolutions from v1:\n     - Replaces sigmoid slack_urgency with *smoothed penalty* (quadratic near zero, linear beyond) for stable gradient & noise resilience\n     - Restores energy-per-second (not per-MI) — aligns with time-constrained DDL optimization and avoids workload bias\n     - Introduces *slack-weighted upward_rank*: rank contribution scaled by 1/(1+|slack|) to preserve criticality under safety margin\n     - Uses *uncertainty-gated waiting boost*: activates only when both slack <= 0 AND uncertainty > median → avoids false starvation alarms\n     - Adds *robust duration-aware energy dominance term*: prioritizes low-energy/low-duration tasks via min_incremental_energy / (duration + eps)^0.5\n     - All normalizations use symmetric IQR with std fallback; all divisions and ops are eps-protected; no infinite/nan values possible.'
    eps = 1e-08
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    
    def robust_normalize(x):
        q1, q3 = np.quantile(x, [0.25, 0.75], method='midpoint')
        iqr = q3 - q1
        if iqr < eps:
            std_val = np.std(x) + eps
            return (x - np.mean(x)) / std_val
        return (x - np.median(x)) / (iqr + eps)
    
    # Smoothed slack penalty: quadratic near zero (stable), linear beyond |slack|=1s → avoids sigmoid saturation & noise amplification
    abs_slack = np.abs(slack)
    slack_penalty = np.where(abs_slack <= 1.0, abs_slack**2, abs_slack)
    
    # Energy efficiency: energy-per-second, clipped and normalized; lower is better → negative weight in score
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_sec = min_incremental_energy / duration
    capped_energy_ps = np.clip(energy_per_sec, 1e-6, 1e6)
    norm_energy_ps = robust_normalize(capped_energy_ps)
    
    # Slack-weighted upward rank: preserves critical-path importance but de-emphasizes it when slack is ample
    slack_scale = 1.0 / (1.0 + abs_slack + eps)
    weighted_rank = upward_rank * slack_scale
    norm_weighted_rank = robust_normalize(weighted_rank)
    
    # Risk-conditional waiting boost: only when truly at risk (slack <= 0 AND high uncertainty) → prevents premature fairness interference
    median_uncert = np.median(uncertainty) if len(uncertainty) > 1 else 0.0
    wait_activation = (slack <= 0) & (uncertainty > median_uncert + eps)
    norm_wait = np.where(wait_activation, robust_normalize(ready_wait_time), 0.0)
    
    # Duration-aware energy dominance: favors tasks with low marginal energy *and* short duration (via sqrt scaling)
    energy_dominance = min_incremental_energy / np.sqrt(duration + eps)
    norm_energy_dom = robust_normalize(np.clip(energy_dominance, 1e-6, 1e6))
    
    # Uncertainty boost: only active under deadline stress (slack <= 0), normalized and capped
    norm_uncertainty = np.where(slack <= 0, robust_normalize(np.clip(uncertainty, 0.0, 1e3)), 0.0)
    
    # Time-cost proxy: sqrt(duration) penalizes long-running tasks moderately
    norm_time_cost = robust_normalize(np.sqrt(duration + eps))
    
    # Final score: smaller = higher priority; weights calibrated to balance feasibility (slack), efficiency (energy), and fairness (wait)
    # Dominant terms: slack_penalty (urgency), norm_energy_ps (efficiency), norm_energy_dom (joint efficiency/duration)
    score = (
        +1.0 * slack_penalty           # Urgency: smooth, bounded, monotonic
        - 0.9 * norm_energy_ps         # Efficiency: reward low energy/sec
        - 0.7 * norm_energy_dom        # Joint efficiency: reward low energy *and* short duration
        + 0.4 * norm_weighted_rank     # Critical path: de-emphasized when slack safe
        + 0.3 * norm_uncertainty       # Risk awareness: only under deadline stress
        - 0.2 * norm_wait              # Starvation guard: strict dual-condition activation
        + 0.1 * norm_time_cost         # Duration moderation
    )
    
    # Ensure finite output: replace NaN/inf with large finite bounds
    return np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
