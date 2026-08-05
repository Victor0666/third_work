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
    Mutated priority rule: replaces exponential deadline penalty with smooth arctan-based risk,
    introduces *critical-energy density* (upward_rank * remaining_work / energy) as primary efficiency signal,
    uses MAD-based normalization for better outlier resilience, and adds latency-aware fairness boost.
    
    Key mutations:
    - Deadline risk modeled via bounded arctan(-slack) → smooth, monotonic, numerically stable near zero & extremes
    - Critical-energy density replaces energy-per-latency: prioritizes high-impact work per joule, aligned with DAG structure
    - All normalizations use robust MAD (median absolute deviation) with epsilon fallback instead of IQR
    - Fairness boost now scales with normalized *latency burden*: (exec + comm) / max(latency), capped at 0.12
    - Uncertainty contributes *only when slack > 0*, scaled by clipped sigmoid of (1 - slack_norm) to emphasize low-margin tasks
    - Weight hierarchy adjusted: deadline risk (3.5) >> critical-energy density (2.0) >> upward_rank (1.2) >> fairness (0.12) >> uncertainty (0.18)
    - Removes remaining_work term — redundant with upward_rank and critical-energy density; avoids double-counting
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
    
    def safe_mad_normalize(x):
        """MAD normalization robust to N=1 and constant arrays; returns zeros if degenerate."""
        if x.size == 1:
            return np.zeros_like(x)
        median_x = np.median(x)
        mad = np.median(np.abs(x - median_x)) + eps
        if mad < eps:
            return np.zeros_like(x)
        normed = (x - median_x) / mad
        return np.clip(normed, -3.0, 3.0)
    
    # Smooth, bounded deadline risk: arctan(-slack) ∈ [-π/2, π/2] → shifted & scaled to [0, 1]
    # Negative slack → positive risk; zero slack → neutral; large positive slack → near-zero risk
    deadline_risk_raw = np.arctan(-slack) + np.pi/2  # maps (-∞, ∞) → (0, π)
    deadline_risk_scaled = deadline_risk_raw / np.pi  # maps to (0, 1)
    deadline_score = safe_mad_normalize(deadline_risk_scaled)
    
    # Critical-energy density: upward_rank * remaining_work / (energy + eps) — captures importance-per-joule
    # Only active when slack > 0 (DDL-safe regime); zero otherwise to avoid rewarding inefficiency under risk
    critical_energy_density = np.where(
        slack > 0,
        (upward_rank * remaining_work) / (min_incremental_energy + eps),
        0.0
    )
    critical_energy_norm = safe_mad_normalize(critical_energy_density)
    
    # Upward rank (critical path importance) always contributes, but gated by slack sign for robustness
    # When slack <= 0, only deadline risk dominates; upward_rank still normalized but weighted lower
    upward_rank_norm = safe_mad_normalize(upward_rank)
    
    # Latency-aware fairness boost: proportional to task's execution+communication burden relative to max
    # Prevents starvation of large-but-non-critical tasks without overriding DDL priority
    total_latency = min_exec_time + min_comm_time + eps
    max_latency = np.max(total_latency) + eps
    latency_ratio = total_latency / max_latency
    fairness_boost = np.clip(latency_ratio * 0.12, 0.0, 0.12)
    
    # Uncertainty contribution: only for slack > 0, and strongest when slack is small (high margin pressure)
    # Sigmoid of (1 - slack_norm) ensures monotonic decay as slack increases
    slack_margin = np.clip(slack, 0.0, None)
    slack_norm = (slack_margin - np.median(slack_margin)) / (np.median(np.abs(slack_margin - np.median(slack_margin))) + eps)
    slack_norm_clipped = np.clip(slack_norm, 0.0, 10.0)
    uncertainty_gated = np.where(
        slack > 0,
        uncertainty * (1.0 / (1.0 + np.exp(-(1.0 - slack_norm_clipped)))),
        0.0
    )
    uncertainty_norm = safe_mad_normalize(uncertainty_gated)
    
    # Final score: smaller = higher priority
    # Deadline risk dominates (positive weight), critical-energy density is negative (favor high density),
    # upward_rank negative (favor high rank), fairness & uncertainty positive (penalize burden/risk)
    score = (
        +3.5 * deadline_score
        - 2.0 * critical_energy_norm
        - 1.2 * upward_rank_norm
        + 0.12 * fairness_boost
        + 0.18 * uncertainty_norm
    )
    
    return np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
