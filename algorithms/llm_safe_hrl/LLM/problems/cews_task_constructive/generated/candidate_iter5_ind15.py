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
    Hybrid priority rule: combines Parent 2's arctan deadline risk and critical-energy density
    with Parent 1's slack-conditioned aging and robust risk amplification, plus novel
    latency-aware fairness and uncertainty gating.
    
    Key improvements:
    - Uses arctan-based deadline risk (smooth, stable, monotonic) from Parent 2
    - Retains critical-energy density (upward_rank * remaining_work / energy) as primary efficiency signal
    - Introduces *slack-conditioned aging* (tanh-based) only when slack < 5.0s to prevent starvation under pressure
    - Applies *risk-amplified deadline score*: multiplies normalized deadline risk by sigmoid(uncertainty) only when slack < 0
    - Adds *latency fairness boost*: proportional to (exec + comm) / max(exec+comm), capped at 0.1, improving load balancing
    - Uses MAD normalization with explicit degenerate-case handling and clipping (-3,3)
    - All operations guarded against division-by-zero, NaN, and inf; final score bounded and sanitized
    """
    eps = 1e-8
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    
    def safe_mad_normalize(x):
        """Robust MAD normalization: handles N=1, constant arrays, and outliers."""
        if x.size == 0:
            return np.zeros_like(x)
        if x.size == 1:
            return np.zeros_like(x)
        median_x = np.median(x)
        mad = np.median(np.abs(x - median_x)) + eps
        if mad < eps:
            return np.zeros_like(x)
        normed = (x - median_x) / mad
        return np.clip(normed, -3.0, 3.0)
    
    # Deadline risk: smooth, bounded arctan-based penalty (Parent 2)
    # arctan(-slack) ∈ (-π/2, π/2) → shift & scale to [0, 1]
    deadline_risk_raw = np.arctan(-slack) + np.pi / 2
    deadline_risk_scaled = deadline_risk_raw / np.pi
    deadline_score = safe_mad_normalize(deadline_risk_scaled)
    
    # Critical-energy density: high-impact work per joule (Parent 2)
    # Only active when energy > 0 and slack permits (avoid division by zero or negative energy)
    energy_denom = min_incremental_energy + eps
    critical_energy_density = upward_rank * remaining_work / energy_denom
    critical_energy_norm = safe_mad_normalize(critical_energy_density)
    
    # Upward rank normalization (Parent 2)
    upward_rank_norm = safe_mad_normalize(upward_rank)
    
    # Slack-conditioned aging: accelerates priority for long-waiting tasks only when deadline is tight
    # Active only when slack < 5.0 seconds (avoids premature aging for relaxed tasks)
    aging_boost = np.where(
        slack < 5.0,
        np.tanh(0.4 * ready_wait_time / (np.abs(slack) + 1.0)),
        0.0
    )
    
    # Risk amplification: multiply deadline score by uncertainty only when slack < 0 (deadline violation imminent)
    risk_amp = np.where(
        slack < 0.0,
        1.0 + 1.5 * (1.0 / (1.0 + np.exp(-uncertainty + 0.5))),
        1.0
    )
    
    # Latency fairness boost: proportional to task's latency burden, capped at 0.1
    total_latency = min_exec_time + min_comm_time + eps
    max_latency = np.max(total_latency) + eps
    fairness_boost = np.clip(total_latency / max_latency * 0.1, 0.0, 0.1)
    
    # Uncertainty contribution only when slack > 0 and uncertainty is high, scaled by slack margin
    slack_margin = np.clip(slack, 0.0, None)
    slack_norm = (slack_margin - np.median(slack_margin)) / (np.median(np.abs(slack_margin - np.median(slack_margin))) + eps)
    slack_norm_clipped = np.clip(slack_norm, 0.0, 5.0)
    uncertainty_gated = np.where(
        (slack > 0.0) & (uncertainty > 0.1),
        uncertainty * (1.0 / (1.0 + np.exp(-(1.0 - slack_norm_clipped)))),
        0.0
    )
    uncertainty_norm = safe_mad_normalize(uncertainty_gated)
    
    # Final weighted score: deadline risk dominates, critical density strongly negative (favor), aging positive (penalty)
    # Weights tuned to reflect DDL-hardness first, then efficiency, then fairness and risk awareness
    score = (
        4.0 * deadline_score * risk_amp  # Hard deadline safety first
        - 2.2 * critical_energy_norm     # Maximize critical work per joule
        - 1.1 * upward_rank_norm         # Preserve critical-path ordering
        + 0.08 * aging_boost             # Prevent starvation under tight slack
        + 0.1 * fairness_boost           # Balance latency-heavy tasks
        + 0.15 * uncertainty_norm       # Add caution for uncertain-but-relaxed tasks
    )
    
    # Bounded and sanitized output
    score = np.clip(score, -1e12, 1e12)
    return np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
