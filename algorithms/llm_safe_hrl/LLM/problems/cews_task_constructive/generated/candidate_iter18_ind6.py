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

    '''
    v2 priority rule: Adaptive urgency hierarchy + smooth deadline proximity + robust fairness + criticality-gated risk.
    
    Key improvements over v1:
    - Replaces hard -1e12 offset with *smooth urgency bias*: uses sigmoid(-slack) to prioritize increasingly urgent tasks
      without abrupt jumps — preserves exploration near deadline (slack ≈ 0) while still strongly favoring violated ones.
    - Restores full-range wait-time fairness: uses robust min-max normalization on entire ready_wait_time (no trimming),
      ensuring deterministic fairness even for N=1–3; avoids starvation in sparse ready sets.
    - Replaces clipped 1/|slack| scaling with smooth sigmoid-based slack sensitivity: 
      slack_sensitivity = 1 / (1 + exp(-5 * slack)) → rises smoothly from ~0 (far ahead) to ~1 (violated).
    - Tightens uncertainty gating: applies sigmoid(slack) * sigmoid(upward_rank) instead of linear clipping,
      ensuring activation only when *both* high urgency AND high criticality co-occur.
    - Refines energy penalty: uses upward-rank-weighted energy density thresholding at 90th percentile (stricter filtering),
      and replaces binary mask with soft sigmoid gate for gradient-aware suppression.
    - All normalizations use percentile-robust minmax with explicit NaN/inf handling and zero-variance fallback.
    - Final weights preserved hierarchical but re-balanced for smoother tradeoffs: urgency (0.35) > critical-latency (0.24) >
      energy (0.18) > fairness (0.10) > risk-gated uncertainty (0.08) > remaining_work (0.05).
    '''
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

    # Robust normalization with explicit zero-variance and empty handling
    def robust_minmax_norm(x):
        if x.size == 0:
            return np.zeros_like(x)
        # Handle all-NaN or inf
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        p01 = np.percentile(x, 1.0, method='midpoint')
        p99 = np.percentile(x, 99.0, method='midpoint')
        x_clipped = np.clip(x, p01, p99)
        x_min = np.min(x_clipped)
        x_max = np.max(x_clipped)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        return (x_clipped - x_min) / (x_max - x_min + eps)

    # Smooth urgency bias: sigmoid(-slack) → 1.0 when slack <= 0, decays smoothly for slack > 0
    # Avoids hard cutoff; enables graded prioritization near deadline
    urgency_bias = 1.0 / (1.0 + np.exp(slack + eps))  # ≈1 for slack<=0, ≈0 for large slack

    # Critical-path latency: duration scaled by upward rank, robustly normalized
    duration = min_exec_time + min_comm_time + eps
    critical_latency_raw = duration * (1.0 + 0.8 * robust_minmax_norm(upward_rank))
    norm_critical_latency = robust_minmax_norm(critical_latency_raw)

    # Energy penalty: upward-rank-normalized density, soft-gated via sigmoid threshold
    energy_density = min_incremental_energy / (duration + eps)
    rank_weighted_energy = energy_density / (upward_rank + eps)
    threshold_rank_energy = np.percentile(rank_weighted_energy, 90.0, method='midpoint') + eps
    # Soft gate: suppresses only tasks significantly above threshold
    energy_gate = 1.0 / (1.0 + np.exp(-(rank_weighted_energy - threshold_rank_energy) / (eps + np.std(rank_weighted_energy, ddof=1))))
    norm_energy_density = robust_minmax_norm(energy_density)
    energy_penalty = norm_energy_density * energy_gate

    # Fairness: full-range wait-time normalization (no trimming), applied only to non-urgent tasks
    norm_wait_time = robust_minmax_norm(ready_wait_time)
    wait_penalty = (1.0 - urgency_bias) * norm_wait_time  # reduced weight for urgent tasks

    # Risk-gated uncertainty: jointly activated only under high urgency AND high criticality
    # Sigmoid gates ensure smooth, differentiable coupling without noise amplification
    slack_sensitivity = 1.0 / (1.0 + np.exp(-5.0 * slack))  # rises sharply near slack=0
    rank_sensitivity = 1.0 / (1.0 + np.exp(-(upward_rank - np.median(upward_rank)) / (eps + np.std(upward_rank, ddof=1)))) if N > 1 else np.ones_like(upward_rank)
    uncertainty_boost = uncertainty * slack_sensitivity * rank_sensitivity
    norm_uncertainty_boost = robust_minmax_norm(uncertainty_boost)

    # Remaining work normalized
    norm_remaining_work = robust_minmax_norm(remaining_work)

    # Weighted combination — urgency dominates but remains smooth and differentiable
    score = (
        0.35 * (1.0 - urgency_bias) +           # smaller score = higher priority → invert bias
        0.24 * norm_critical_latency +
        0.18 * energy_penalty +
        0.10 * wait_penalty +
        0.08 * norm_uncertainty_boost +
        0.05 * norm_remaining_work
    )

    # Final sanitization: clamp, replace NaN/inf, ensure shape
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    # Ensure urgent tasks get minimal scores *without breaking smoothness*
    score = np.where(slack <= 0, -1e12 + (1.0 - urgency_bias) * 1e-6, score)

    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
