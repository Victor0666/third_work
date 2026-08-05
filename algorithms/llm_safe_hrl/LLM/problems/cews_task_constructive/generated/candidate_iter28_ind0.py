import numpy as np

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
    v2 priority rule: Hard urgency dominance + tanh-based slack urgency + 
    work-normalized energy density + uncertainty-gated fairness + robust 5–95% scaling.
    
    Key mutations:
    - Replaces step-based urgent flag with smooth tanh(-slack/τ) urgency, avoiding hard cliffs while preserving dominance near deadline.
    - Uses *remaining_work*-normalized energy density (not duration) to penalize high-energy tasks per MI of critical path work.
    - Introduces uncertainty-gated fairness: starvation relief only activates when uncertainty > median AND slack > 0 (non-urgent but risky).
    - Applies 5–95% percentile min-max instead of 1–99% for tighter dynamic range and better small-N stability.
    - Removes z-score-based wait scaling; uses direct robust minmax on ready_wait_time × (1 + uncertainty), enhancing starvation resilience under volatility.
    - Energy penalty now gated by both upward_rank *and* normalized slack proximity (|rel_slack| < 0.2), sharpening critical-path focus.
    - All divisions guarded; NaN/inf replaced deterministically; no side effects.
    """
    eps = 1e-8
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

    def robust_minmax_norm_595(x):
        if x.size == 0:
            return np.zeros_like(x)
        p05 = np.percentile(x, 5.0)
        p95 = np.percentile(x, 95.0)
        x_clipped = np.clip(x, p05, p95)
        x_min = np.min(x_clipped)
        x_max = np.max(x_clipped)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        return (x_clipped - x_min) / (x_max - x_min + eps)

    # Smooth urgency: tanh(-slack / (avg_duration + eps)) → -1 (urgent) to 0 (safe)
    duration = min_exec_time + min_comm_time + eps
    avg_duration = np.mean(duration) if N > 0 else eps
    tau = np.maximum(avg_duration * 0.5, eps)  # adaptive time constant
    urgency_raw = -slack / tau
    urgency = np.tanh(urgency_raw)  # [-1.0, 0.0] for slack <= 0 → increasingly negative; positive slack → near zero

    # Critical path latency: weighted by upward_rank, normalized
    critical_latency = duration * (1.0 + 0.6 * robust_minmax_norm_595(upward_rank))
    norm_critical_latency = robust_minmax_norm_595(critical_latency)

    # Energy density per MI of remaining work (not per second) → prioritizes energy efficiency on high-work critical paths
    energy_per_work = np.divide(min_incremental_energy, remaining_work + eps,
                                out=np.zeros_like(min_incremental_energy),
                                where=remaining_work + eps != 0)
    norm_energy_per_work = robust_minmax_norm_595(energy_per_work)

    # Tight-slack gating: only penalize energy when both rank is high *and* relative slack is tight
    rel_slack = np.divide(slack, duration, out=np.zeros_like(slack), where=duration != 0)
    rel_slack = np.where(np.isfinite(rel_slack), rel_slack, 0.0)
    abs_rel_slack = np.abs(rel_slack) + eps
    tight_slack_mask = (abs_rel_slack < 0.2).astype(float)
    rank_threshold = np.percentile(upward_rank, 80.0) + eps
    high_rank_mask = (upward_rank > rank_threshold).astype(float)
    energy_penalty_mask = tight_slack_mask * high_rank_mask
    energy_penalty = norm_energy_per_work * energy_penalty_mask

    # Fairness: wait-based rescue gated by *both* uncertainty and non-urgency
    # Only activate fairness when task is not urgent (slack > 0) AND uncertainty is above median
    median_uncert = np.median(uncertainty) if N > 0 else 0.0
    fairness_gate = ((slack > 0.0) & (uncertainty > median_uncert + eps)).astype(float)
    wait_signal = ready_wait_time * (1.0 + uncertainty)  # amplify wait under high risk
    norm_wait_signal = robust_minmax_norm_595(wait_signal)
    wait_penalty = fairness_gate * norm_wait_signal

    # Uncertainty coupling: boosts priority only for high-rank, low-slack tasks — avoids noise on trivial nodes
    rank_slack_coupling = robust_minmax_norm_595(upward_rank) * (1.0 - np.clip(rel_slack, 0.0, 1.0))
    uncertainty_boost = uncertainty * rank_slack_coupling
    norm_uncertainty_boost = robust_minmax_norm_595(uncertainty_boost)

    # Base score anchored at urgency; others add penalty (higher = lower priority)
    score = urgency.copy()  # most urgent → most negative → highest priority
    score += 0.30 * norm_critical_latency
    score += 0.25 * energy_penalty
    score += 0.15 * wait_penalty
    score += 0.12 * norm_uncertainty_boost
    score += 0.08 * robust_minmax_norm_595(remaining_work)  # slight bias toward larger workloads when otherwise equal

    # Final safeguard: clamp extreme values, replace NaN/inf
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)

    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
