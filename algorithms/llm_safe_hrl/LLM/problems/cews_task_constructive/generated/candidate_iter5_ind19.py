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
    Hybrid priority rule: hard deadline safety first, critical-path fidelity second,
    energy-latency efficiency third — with robust normalization, bounded fairness,
    and uncertainty-aware margin decay.
    
    Key improvements:
    - Combines Parent 2's strict binary deadline gating (exponential for slack < 0) 
      with Parent 1's arctan-based smooth penalty for slack >= 0 → avoids overflow 
      while preserving urgency gradient near zero.
    - Uses critical-energy efficiency (upward_rank * remaining_work / energy) 
      only when slack > 0, normalized via MAD for outlier robustness (Parent 1 strength).
    - Introduces *latency-safety inflation*: inflates communication time only when 
      slack <= 0 AND uncertainty > 0.5, preventing premature aggressive scheduling 
      under high risk.
    - Fairness uses percentile-scaled linear boost (0–0.1) as in Parent 1 for monotonicity 
      and interpretable starvation ceiling.
    - All normalizations are degenerate-safe: MAD for critical terms, IQR for latency/uncertainty.
    - Final weights enforce objective hierarchy: deadline (3.5) >> critical-energy (2.0) >> 
      upward_rank (1.2) >> fairness (0.12) >> uncertainty-latency (0.25) >> work (0.08).
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
        med = np.median(x)
        mad = np.median(np.abs(x - med)) + eps
        if mad < eps:
            return np.zeros_like(x)
        normed = (x - med) / mad
        return np.clip(normed, -4.0, 4.0)
    
    def safe_iqr_normalize(x):
        """IQR normalization robust to N=1 and constant arrays; returns zeros if degenerate."""
        if x.size == 1:
            return np.zeros_like(x)
        q1 = np.percentile(x, 25)
        q3 = np.percentile(x, 75)
        iqr = q3 - q1 + eps
        if iqr < eps:
            return np.zeros_like(x)
        normed = (x - q1) / iqr
        return np.clip(normed, -3.0, 3.0)
    
    # Deadline risk: exponential penalty for slack < 0, smooth arctan for slack >= 0
    deadline_risk_raw = np.where(
        slack < 0,
        np.exp(np.clip(-slack, 0, 20)) - 1.0,
        np.arctan(-slack)  # bounded, smooth near zero
    )
    deadline_score = safe_iqr_normalize(deadline_risk_raw)
    
    # Latency-safety inflation: only when high uncertainty AND negative/zero slack
    comm_inflated = np.where(
        (slack <= 0) & (uncertainty > 0.5),
        min_comm_time * (1.0 + np.clip(uncertainty, 0.0, 0.8)),
        min_comm_time
    )
    total_latency = min_exec_time + comm_inflated + eps
    
    # Critical-energy efficiency: only active when slack > 0
    crit_energy_raw = np.where(
        slack > 0,
        (upward_rank * remaining_work + eps) / (min_incremental_energy + eps),
        0.0
    )
    crit_energy_norm = safe_mad_normalize(crit_energy_raw)
    
    # Upward rank gated by slack > 0 (strict criticality enforcement)
    upward_rank_active = np.where(slack > 0, upward_rank, 0.0)
    upward_rank_norm = safe_mad_normalize(upward_rank_active)
    
    # Fairness: linear percentile-scaled boost [0, 0.1], more interpretable than sqrt
    if ready_wait_time.size > 1:
        p95 = np.percentile(ready_wait_time, 95)
        wait_boost = np.clip(ready_wait_time / (p95 + eps), 0.0, 1.0) * 0.1
    else:
        wait_boost = np.full_like(ready_wait_time, 0.05, dtype=float)
    
    # Uncertainty contribution: only when slack > 0, scaled by normalized slack margin
    slack_margin = np.clip(slack, 0.0, None)
    slack_normed = slack_margin / (np.max(slack_margin + eps) + eps)
    uncertainty_gated = np.where(slack > 0, uncertainty * slack_normed, 0.0)
    uncertainty_norm = safe_iqr_normalize(uncertainty_gated)
    
    # Remaining work normalized separately (lower weight, supports load balancing)
    remaining_work_norm = safe_mad_normalize(remaining_work)
    
    # Final weighted score: smaller = higher priority
    score = (
        +3.5 * deadline_score
        - 2.0 * crit_energy_norm
        - 1.2 * upward_rank_norm
        + 0.12 * wait_boost
        + 0.25 * uncertainty_norm
        + 0.08 * remaining_work_norm
    )
    
    return np.nan_to_num(score, nan=1000000000000.0, posinf=1000000000000.0, neginf=-1000000000000.0)
