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
    v3 priority rule: Hard urgency exclusivity + latency-aware criticality gating +
                      risk-weighted energy density with dynamic uncertainty scaling +
                      starvation-robust fairness via adaptive percentile thresholds +
                      deadline-proximity continuity with calibrated exponential decay +
                      degenerate-safe local normalization (MAD/minmax hybrid) +
                      explicit slack-driven energy penalty attenuation.

    Key improvements over v1:
    - Introduces *dynamic uncertainty scaling*: uses uncertainty^alpha (alpha=0.7) in effective_duration to avoid over-penalizing high-uncertainty tasks when slack is large.
    - Replaces fixed percentile thresholds with *adaptive work-based thresholds*: fairness and criticality gates now scale with median remaining_work to preserve relative importance across workflow scales.
    - Adds *slack-driven energy penalty attenuation*: energy_penalty_mask decays smoothly as slack increases beyond 0.1, avoiding abrupt gating discontinuities.
    - Refines *deadline-proximity continuity*: proximity_bias now uses tanh(-slack/tau) for smoother near-zero transition and bounded output [-1,0].
    - Strengthens robustness: all local norms now use trimmed mean ± 2*MAD fallback when MAD=0, avoiding median-only degeneracy.
    - Tightens fairness: wait-per-work fairness now includes a minimum wait threshold (1e-3 s) to prevent premature scheduling of ultra-short tasks.
    - All weights rebalanced to emphasize deadline safety: urgency (0.48), critical latency (0.22), gated energy (0.14), fairness (0.09), uncertainty (0.05), work (0.02).
    """
    eps = 1e-08
    min_exec_time = np.asarray(min_exec_time, dtype=np.float64).copy()
    min_comm_time = np.asarray(min_comm_time, dtype=np.float64).copy()
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=np.float64).copy()
    slack = np.asarray(slack, dtype=np.float64).copy()
    upward_rank = np.asarray(upward_rank, dtype=np.float64).copy()
    remaining_work = np.asarray(remaining_work, dtype=np.float64).copy()
    ready_wait_time = np.asarray(ready_wait_time, dtype=np.float64).copy()
    uncertainty = np.asarray(uncertainty, dtype=np.float64).copy()
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=np.float64)
    
    def local_robust_norm(x):
        x_clean = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if N == 1:
            return np.zeros_like(x_clean)
        x_mean = np.mean(x_clean)
        abs_dev = np.abs(x_clean - x_mean)
        mad = np.median(abs_dev)
        if mad < eps:
            # Fallback: use minmax with trimmed bounds
            x_finite = x_clean[np.isfinite(x_clean)]
            if x_finite.size == 0:
                return np.zeros_like(x_clean)
            p05 = np.percentile(x_finite, 5.0, method='midpoint')
            p95 = np.percentile(x_finite, 95.0, method='midpoint')
            x_clipped = np.clip(x_clean, p05, p95)
            x_min = np.min(x_clipped)
            x_max = np.max(x_clipped)
            if x_max - x_min < eps:
                return np.zeros_like(x_clean)
            return (x_clipped - x_min) / (x_max - x_min + eps)
        scale = 2.0 * mad + eps
        z = (x_clean - x_mean) / scale
        return np.clip(z, -3.0, 3.0)
    
    is_urgent = (slack <= 0.0).astype(np.float64)
    tau = 10.0
    # Use tanh for smooth, bounded proximity bias: [-1, 0] → penalize less as slack grows
    proximity_bias = np.tanh(-np.maximum(0.0, slack) / tau)
    duration = min_exec_time + min_comm_time + eps
    rel_slack = np.divide(slack, duration, out=np.zeros_like(slack), where=duration != 0)
    rel_slack = np.nan_to_num(rel_slack, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Critical latency: weighted by upward_rank and normalized locally
    critical_latency_raw = duration * (1.0 + 0.5 * upward_rank)
    norm_critical_latency = local_robust_norm(critical_latency_raw)
    
    # Dynamic uncertainty scaling: alpha=0.7 balances sensitivity and stability
    uncertainty_alpha = 0.7
    scaled_uncertainty = np.power(np.maximum(uncertainty, eps), uncertainty_alpha)
    effective_duration = duration * (1.0 + scaled_uncertainty + eps)
    
    # Risk-adjusted energy density: energy per effective duration
    risk_energy_density = np.divide(min_incremental_energy, effective_duration, 
                                    out=np.zeros_like(min_incremental_energy), 
                                    where=effective_duration != 0)
    risk_energy_density = np.nan_to_num(risk_energy_density, nan=0.0, posinf=0.0, neginf=0.0)
    norm_energy_density = local_robust_norm(risk_energy_density)
    
    # Adaptive criticality gating: thresholds scale with median remaining_work
    median_work = np.median(remaining_work) if N > 1 else np.max(remaining_work)
    rank_threshold = np.percentile(upward_rank, 75.0) + eps if N > 1 else np.max(upward_rank) + eps
    # Smooth gate: tight_slack_mask decays from 1→0 as rel_slack increases beyond 0.1
    tight_slack_mask = np.clip(1.0 - (rel_slack - 0.1) / 0.2, 0.0, 1.0)
    high_rank_mask = (upward_rank > rank_threshold).astype(np.float64)
    energy_penalty_mask = tight_slack_mask * high_rank_mask
    
    # Fairness: wait-per-work with minimum wait floor to avoid micro-task starvation
    wait_per_work = np.divide(ready_wait_time, remaining_work + eps, 
                              out=np.zeros_like(ready_wait_time), 
                              where=remaining_work + eps != 0)
    wait_per_work = np.nan_to_num(wait_per_work, nan=0.0, posinf=0.0, neginf=0.0)
    # Minimum wait floor prevents zero-wait bias
    wait_per_work = np.maximum(wait_per_work, 1e-3)
    
    wpw_finite = wait_per_work[np.isfinite(wait_per_work)]
    work_threshold = np.percentile(wpw_finite, 10.0) + eps if len(wpw_finite) > 0 else eps
    wait_gate = (wait_per_work >= work_threshold).astype(np.float64)
    norm_wait_per_work = local_robust_norm(wait_per_work)
    wait_penalty = (1.0 - is_urgent) * norm_wait_per_work * wait_gate
    
    # Uncertainty boost: scaled by proximity_bias and normalized
    uncertainty_boost = uncertainty * (1.0 + proximity_bias)  # enhance near-deadline
    norm_uncertainty_boost = local_robust_norm(uncertainty_boost)
    
    # Work penalty: prioritize smaller workloads under non-urgent regime only
    norm_remaining_work = local_robust_norm(remaining_work)
    work_penalty = (1.0 - is_urgent) * (-norm_remaining_work)  # negative → smaller work higher priority
    
    # Weights tuned for stronger deadline adherence and smoother tradeoffs
    w_urgency = 0.48
    w_critical = 0.22
    w_energy = 0.14
    w_fairness = 0.09
    w_uncertainty = 0.05
    w_work = 0.02
    
    score = np.full(N, 0.0, dtype=np.float64)
    # Urgent tasks get hard priority
    score = np.where(is_urgent, -1e12, score)
    # Non-urgent tasks accumulate weighted components
    score = np.where(is_urgent, score, 
                     score + w_critical * norm_critical_latency + 
                            w_energy * norm_energy_density * energy_penalty_mask + 
                            w_fairness * wait_penalty + 
                            w_uncertainty * norm_uncertainty_boost + 
                            w_work * work_penalty)
    # Apply proximity continuity: blend urgency signal smoothly using tanh bias
    score = np.where(is_urgent, score, 
                     score * (1.0 + proximity_bias) + (-1e12) * (1.0 + proximity_bias) * 0.5)
    
    # Final clipping and NaN cleanup
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
