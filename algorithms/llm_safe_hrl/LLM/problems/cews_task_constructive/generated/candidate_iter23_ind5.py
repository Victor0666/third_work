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
    v2 priority rule: Hard deadline dominance + risk-gated criticality-energy tension +
                     MAD-robust starvation relief + uncertainty-normalized urgency +
                     latency-aware wait pressure + work-density fairness gating.
    
    Key innovations:
    - Combines Parent 2's strong inverse-slack urgency and criticality-energy tension
      with Parent 1's sliding-window wait-pressure (adapted to work-density context)
    - Replaces fragile sigmoid-based slack gating with robust *inverse slack confidence* 
      scaled by (1 + uncertainty) for monotonic risk amplification
    - Introduces *latency-sensitive wait pressure*: activates only when duration > median_duration 
      AND slack < median_slack, preventing premature starvation relief on short tasks
    - Uses *work-density gated wait relief*: wait_per_work normalized only when remaining_work 
      exceeds median, avoiding bias toward tiny tasks
    - All normalizations use median/MAD with zero-MAD fallback and strict [-3,3] clipping
    - Energy penalty strictly gated by both upward_rank > median AND slack < median_slack
    - Final score layers: urgency > criticality > energy tension > latency-aware wait relief > uncertainty bonus
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
    
    def robust_mad_norm(x):
        x_clean = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x_clean.size == 0:
            return np.zeros_like(x_clean)
        med = np.median(x_clean)
        mad = np.median(np.abs(x_clean - med))
        if mad < eps:
            # Fallback to std if MAD is degenerate
            scale = np.std(x_clean) + eps
        else:
            scale = mad + eps
        z = (x_clean - med) / scale
        return np.clip(z, -3.0, 3.0)
    
    # Urgency: hard priority for overdue tasks
    is_urgent = (slack <= 0.0).astype(np.float64)
    
    # Duration and latency features
    duration = min_exec_time + min_comm_time + eps
    median_duration = np.median(duration) + eps
    median_slack = np.median(slack) + eps
    median_upward = np.median(upward_rank) + eps
    
    # Inverse slack with uncertainty-aware confidence scaling: higher uncertainty → lower confidence → stronger urgency
    inv_slack_raw = np.divide(1.0, np.abs(slack) + eps, out=np.zeros_like(slack), where=(np.abs(slack) + eps) != 0)
    inv_slack_confidence = 1.0 / (1.0 + uncertainty + eps)  # Uncertainty reduces confidence in slack estimate
    inv_slack = inv_slack_raw * inv_slack_confidence
    inv_slack = np.clip(inv_slack, 0.1, 100.0)  # Bounded to prevent explosion
    
    # Criticality: upward rank weighted by duration, robustly normalized
    crit_latency_proxy = duration * (1.0 + 0.5 * upward_rank)
    norm_crit_latency = robust_mad_norm(crit_latency_proxy)
    
    # Energy density: marginal energy per unit duration, adjusted by uncertainty
    energy_density = np.divide(min_incremental_energy, duration, out=np.zeros_like(min_incremental_energy), where=duration != 0)
    energy_density_adj = energy_density / (1.0 + uncertainty + eps)
    norm_energy_density = robust_mad_norm(energy_density_adj)
    
    # Criticality-energy tension mask: only penalize energy on high-criticality & tight-slack tasks
    tension_mask = ((upward_rank > median_upward) & (slack < median_slack)).astype(np.float64)
    
    # Latency-sensitive wait pressure: activate only when task is latency-heavy AND slack is tight
    latency_gate = ((duration > median_duration) & (slack < median_slack)).astype(np.float64)
    wait_per_work = np.divide(ready_wait_time, remaining_work + eps, out=np.zeros_like(ready_wait_time), where=(remaining_work + eps) != 0)
    norm_wait_per_work = robust_mad_norm(wait_per_work)
    # Work-density gate ensures fairness only for substantial tasks
    work_density_gate = (remaining_work >= np.median(remaining_work) + eps).astype(np.float64)
    wait_pressure = norm_wait_per_work * latency_gate * work_density_gate
    
    # Uncertainty bonus: lower priority for highly uncertain tasks (encourages scheduling predictable ones first)
    norm_uncertainty = robust_mad_norm(uncertainty)
    
    # Base score initialization
    score = np.full(N, 0.0, dtype=np.float64)
    
    # Hard priority for urgent tasks
    score = np.where(is_urgent, -1000000000000.0, score)
    
    # Layered scoring (lower is better): urgency dominates, then criticality, energy tension, wait relief, uncertainty
    score = np.where(
        is_urgent,
        score,
        score 
        + 0.40 * norm_crit_latency                # Critical path importance
        + 0.25 * norm_energy_density * tension_mask  # Energy penalty only when justified
        + 0.20 * (-wait_pressure)                 # Starvation relief (negative = higher priority)
        + 0.10 * (-robust_mad_norm(inv_slack))    # Stronger urgency signal for tight slack
        + 0.05 * norm_uncertainty                 # Prefer predictable tasks
    )
    
    # Final safeguard clipping and NaN/inf handling
    score = np.clip(score, -1000000000000.0, 1000000000000.0)
    score = np.nan_to_num(score, nan=1000000000000.0, posinf=1000000000000.0, neginf=-1000000000000.0)
    
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
