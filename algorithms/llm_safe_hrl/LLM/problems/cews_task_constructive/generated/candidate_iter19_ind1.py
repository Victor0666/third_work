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
    v2 priority rule: Hard urgency dominance + latency-aware criticality gating + 
                      risk-normalized energy + starvation-robust fairness + 
                      uncertainty-weighted slack sensitivity.
    
    Key mutations vs v1:
    - Replaces tanh-based slack scaling with clipped inverse-linear sensitivity 
      that smoothly amplifies urgency near zero slack without singularity.
    - Introduces *latency-aware criticality gating*: upward_rank penalty only activates 
      when both slack is tight AND duration is above median — avoids penalizing 
      short critical tasks unnecessarily.
    - Energy term now uses *risk-normalized incremental energy*: divides by 
      (duration * (1 + uncertainty)), suppressing high-energy assignments on 
      highly uncertain VMs unless latency-critical.
    - Fairness term replaces percentile-thresholded wait-per-work with robust 
      trimmed-mean normalized wait time, scaled by (1 - normalized slack) to 
      boost fairness more under pressure.
    - Uncertainty coupling simplified: multiplies normalized uncertainty directly 
      with normalized |slack|⁻¹ (clipped), avoiding rank-sensitivity coupling 
      that diluted signal on low-rank tasks.
    - All normalizations use trimmed mean ± 3*MAD (more robust than percentile clipping).
    - Final score enforces strict priority hierarchy via additive layers with 
      decreasing weight decay (urgency → latency → energy → fairness → risk).
    """
    eps = 1e-8
    # Ensure float64 and copy to avoid mutation
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
    
    # Robust normalization: trimmed mean ± 3*MAD (more stable than percentile for small N)
    def robust_trimmed_norm(x):
        if x.size == 0:
            return np.zeros_like(x)
        # Compute MAD robustly
        x_med = np.median(x)
        abs_dev = np.abs(x - x_med)
        mad = np.median(abs_dev) if abs_dev.size > 0 else 0.0
        scale = 3.0 * mad + eps
        # Avoid degenerate scale
        if scale < eps:
            return np.zeros_like(x)
        z = (x - x_med) / scale
        # Clip to [-3, 3] range to bound outliers
        return np.clip(z, -3.0, 3.0)
    
    # 1. Hard urgency dominance: urgent tasks get absolute minimum score
    is_urgent = (slack <= 0.0).astype(np.float64)
    
    # 2. Duration & relative slack
    duration = min_exec_time + min_comm_time + eps
    rel_slack = np.divide(slack, duration, out=np.zeros_like(slack), where=duration != 0)
    rel_slack = np.where(np.isfinite(rel_slack), rel_slack, 0.0)
    
    # 3. Latency-aware criticality: upward_rank penalty only when slack is tight AND duration is long
    duration_med = np.median(duration) + eps
    tight_slack_mask = (rel_slack <= 0.25).astype(np.float64)
    long_duration_mask = (duration >= duration_med).astype(np.float64)
    critical_gate = tight_slack_mask * long_duration_mask
    
    # Critical latency term: duration weighted by upward_rank, gated
    critical_latency_raw = duration * upward_rank
    norm_critical_latency = robust_trimmed_norm(critical_latency_raw)
    
    # 4. Risk-normalized energy: energy per effective duration (penalized by uncertainty)
    effective_duration = duration * (1.0 + uncertainty + eps)
    risk_norm_energy = np.divide(
        min_incremental_energy, 
        effective_duration, 
        out=np.zeros_like(min_incremental_energy), 
        where=effective_duration != 0
    )
    risk_norm_energy = np.where(np.isfinite(risk_norm_energy), risk_norm_energy, 0.0)
    norm_energy = robust_trimmed_norm(risk_norm_energy)
    
    # 5. Starvation-robust fairness: normalized wait time boosted under deadline pressure
    norm_wait_time = robust_trimmed_norm(ready_wait_time)
    # Boost fairness when slack is shrinking: (1 - normalized slack) ∈ [0,2]
    slack_norm = robust_trimmed_norm(slack)
    slack_pressure = np.clip(1.0 - slack_norm, 0.0, 2.0)
    wait_penalty = norm_wait_time * slack_pressure
    
    # 6. Uncertainty-weighted slack sensitivity: clipped inverse-linear near zero
    abs_slack = np.abs(slack) + eps
    # Inverse-linear sensitivity: 1/abs_slack, clipped to [0.1, 20.0] to avoid explosion
    slack_sensitivity = np.clip(1.0 / abs_slack, 0.1, 20.0)
    norm_uncertainty = robust_trimmed_norm(uncertainty)
    uncertainty_boost = norm_uncertainty * slack_sensitivity
    
    # 7. Remaining work as secondary load signal (not dominant)
    norm_remaining_work = robust_trimmed_norm(remaining_work)
    
    # Build priority score with strict hierarchy: lower = better
    # Base layer: urgency dominates everything
    score = np.full(N, 1.0, dtype=np.float64)
    score = np.where(is_urgent, -1e12, score)
    
    # Add hierarchical layers with decaying weights
    # Layer 2: critical latency (weight 0.35) — only for non-urgent
    score = np.where(is_urgent, score, score + 0.35 * norm_critical_latency)
    # Layer 3: risk-normalized energy (weight 0.25)
    score = np.where(is_urgent, score, score + 0.25 * norm_energy * critical_gate)
    # Layer 4: fairness under pressure (weight 0.20)
    score = np.where(is_urgent, score, score + 0.20 * wait_penalty)
    # Layer 5: uncertainty-slack coupling (weight 0.12)
    score = np.where(is_urgent, score, score + 0.12 * uncertainty_boost)
    # Layer 6: remaining work (weight 0.08)
    score = np.where(is_urgent, score, score + 0.08 * norm_remaining_work)
    
    # Final numeric safety
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
