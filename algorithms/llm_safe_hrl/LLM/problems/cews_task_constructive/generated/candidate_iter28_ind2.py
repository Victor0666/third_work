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
    v2 priority rule: Hard urgency dominance + slack-proximity tanh scaling + 
                      criticality-gated energy density + starvation-aware wait rescue +
                      uncertainty-weighted deadline proximity.
    
    Key mutations vs v1:
    - Replaces step-like urgency with smooth tanh-based slack pressure (no discontinuities)
    - Uses *relative slack* (slack / (exec+comm)) as primary urgency axis, normalized via tanh(-x) → [0,1]
    - Energy penalty now gated by *both* upward_rank *and* normalized slack proximity (tanh(-rel_slack))
    - Starvation relief uses median + 0.75×IQR (more robust than percentile) and scales wait-time by work density
    - Uncertainty term multiplied by tanh(|rel_slack|) to suppress noise on non-urgent tasks
    - All components normalized via 5–95% clipping (more stable for small N than 1–99%)
    - Final score combines urgency base, critical path cost, energy penalty, fairness boost, and risk modulation
    - No hardcoded large constants: all terms bounded in [0,1] before weighting
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

    def robust_5_95_norm(x):
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

    # Duration baseline (exec + comm), avoid zero
    duration = min_exec_time + min_comm_time + eps

    # Relative slack: negative = urgent; large positive = loose
    rel_slack = np.divide(slack, duration, out=np.zeros_like(slack), where=duration != 0)
    rel_slack = np.where(np.isfinite(rel_slack), rel_slack, 0.0)

    # Smooth urgency: tanh(-rel_slack) → [0,1], steep near 0, saturates at ±3
    # Maps rel_slack <= -3 → ~1.0 (max urgency), rel_slack >= 3 → ~0.0 (low urgency)
    urgency_base = np.tanh(-rel_slack)

    # Critical path latency: duration weighted by upward rank (higher rank = more critical)
    critical_latency = duration * (1.0 + 0.5 * robust_5_95_norm(upward_rank))
    norm_critical_latency = robust_5_95_norm(critical_latency)

    # Energy density: marginal energy per time unit
    energy_density = np.divide(min_incremental_energy, duration, out=np.zeros_like(min_incremental_energy), where=duration != 0)
    norm_energy_density = robust_5_95_norm(energy_density)

    # Energy penalty only activates for high-criticality *and* high-urgency tasks
    # Use tanh(-rel_slack) instead of binary mask for smooth gating
    urgency_gate = np.tanh(-np.clip(rel_slack, -2.0, 2.0))  # bounded to avoid overflow
    rank_gate = robust_5_95_norm(upward_rank)
    energy_penalty = norm_energy_density * urgency_gate * rank_gate

    # Starvation rescue: activate only for high-work tasks with long wait, scaled by work density
    work_density = np.divide(remaining_work, duration + eps, out=np.zeros_like(remaining_work), where=duration + eps != 0)
    work_iqr = np.percentile(remaining_work, 75) - np.percentile(remaining_work, 25) + eps
    work_threshold = np.median(remaining_work) + 0.75 * work_iqr
    wait_gate = (remaining_work >= work_threshold).astype(float)
    # Normalize wait time relative to work density to prioritize high-work/long-wait combos
    wait_score = np.divide(ready_wait_time, work_density + eps, out=np.zeros_like(ready_wait_time), where=work_density + eps != 0)
    norm_wait_score = robust_5_95_norm(wait_score)
    wait_penalty = (1.0 - urgency_base) * norm_wait_score * wait_gate

    # Uncertainty modulation: boost priority only when uncertainty matters (i.e., near deadline)
    # Suppress uncertainty effect when rel_slack is large (loose deadline)
    slack_proximity = np.tanh(np.abs(rel_slack))  # → 1.0 when |rel_slack| large, 0.0 when near zero
    uncertainty_mod = uncertainty * (1.0 - slack_proximity)  # active only near deadline
    norm_uncertainty_mod = robust_5_95_norm(uncertainty_mod)

    # Assemble final score: lower = better
    # Base is urgency (0→1), then add penalties (0→1 each), all weighted
    score = (
        0.45 * urgency_base +
        0.20 * norm_critical_latency +
        0.18 * energy_penalty +
        0.10 * wait_penalty +
        0.07 * norm_uncertainty_mod
    )

    # Ensure finite output and shape compliance
    score = np.nan_to_num(score, nan=1.0, posinf=1.0, neginf=0.0)
    score = np.clip(score, 0.0, 1.0)
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
