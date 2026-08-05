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
                      uncertainty-weighted slack sensitivity + adaptive work-density relief.
    
    Key improvements over parents:
    - Combines Parent 2's robust trimmed normalization and criticality gating with Parent 1's work-density penalty
    - Replaces static weights with dynamic weight allocation based on urgency pressure (slack distribution)
    - Introduces *adaptive work-density relief*: penalizes high energy-per-MI only when slack is non-critical,
      but actively rewards low energy-per-MI for urgent tasks via sign-flipped term
    - Uses clipped inverse slack sensitivity with uncertainty-modulated scaling (Parent 1) AND latency-aware gating (Parent 2)
    - All normalizations use trimmed mean ± 3*MAD for small-N stability; explicit NaN/inf guarding at every stage
    - Final score enforces strict priority hierarchy while adapting component weights to current slack pressure
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

    def robust_trimmed_norm(x):
        if x.size == 0:
            return np.zeros_like(x)
        x_clean = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x_med = np.median(x_clean)
        abs_dev = np.abs(x_clean - x_med)
        mad = np.median(abs_dev) if abs_dev.size > 0 else 0.0
        scale = 3.0 * mad + eps
        if scale < eps:
            return np.zeros_like(x_clean)
        z = (x_clean - x_med) / scale
        return np.clip(z, -3.0, 3.0)

    # Hard urgency dominance: urgent tasks get highest priority (lowest score)
    is_urgent = (slack <= 0.0).astype(np.float64)

    # Duration and normalized slack metrics
    duration = min_exec_time + min_comm_time + eps
    rel_slack = np.divide(slack, duration, out=np.zeros_like(slack), where=duration != 0)
    rel_slack = np.nan_to_num(rel_slack, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Criticality gating: only activate latency penalty for tasks that are both tight-slack AND long-duration
    duration_med = np.median(duration) + eps
    tight_slack_mask = (rel_slack <= 0.25).astype(np.float64)
    long_duration_mask = (duration >= duration_med).astype(np.float64)
    critical_gate = tight_slack_mask * long_duration_mask

    # Critical latency: duration * upward_rank, normalized
    critical_latency_raw = duration * upward_rank
    norm_critical_latency = robust_trimmed_norm(critical_latency_raw)

    # Risk-normalized energy: suppress high-energy assignments on uncertain VMs unless latency-critical
    effective_duration = duration * (1.0 + uncertainty + eps)
    risk_norm_energy = np.divide(min_incremental_energy, effective_duration, 
                                 out=np.zeros_like(min_incremental_energy), 
                                 where=effective_duration != 0)
    risk_norm_energy = np.nan_to_num(risk_norm_energy, nan=0.0, posinf=0.0, neginf=0.0)
    norm_energy = robust_trimmed_norm(risk_norm_energy)

    # Starvation-robust fairness: normalized wait time boosted under slack pressure
    norm_wait_time = robust_trimmed_norm(ready_wait_time)
    slack_norm = robust_trimmed_norm(slack)
    slack_pressure = np.clip(1.0 - slack_norm, 0.0, 2.0)  # higher pressure → stronger fairness boost
    wait_penalty = norm_wait_time * slack_pressure

    # Uncertainty-weighted slack sensitivity: clipped inverse |slack| modulated by uncertainty
    abs_slack = np.abs(slack) + eps
    slack_sensitivity = np.clip(1.0 / abs_slack, 0.1, 20.0)
    norm_uncertainty = robust_trimmed_norm(uncertainty)
    uncertainty_boost = norm_uncertainty * slack_sensitivity

    # Adaptive work-density relief: energy per MI, but with sign flip for urgent tasks
    # For non-urgent: penalize high energy/MI (wasteful)
    # For urgent: reward low energy/MI (favor efficient execution under deadline pressure)
    energy_per_mi = np.divide(min_incremental_energy, remaining_work + eps, 
                              out=np.zeros_like(min_incremental_energy), 
                              where=remaining_work + eps != 0)
    energy_per_mi = np.nan_to_num(energy_per_mi, nan=0.0, posinf=0.0, neginf=0.0)
    norm_energy_per_mi = robust_trimmed_norm(energy_per_mi)
    work_density_term = (1.0 - is_urgent) * norm_energy_per_mi - is_urgent * norm_energy_per_mi

    # Dynamic weight allocation based on global slack pressure
    # When slack is globally tight (many negative/low slack), increase urgency & latency weights
    slack_mean = np.mean(slack)
    slack_std = np.std(slack) + eps
    slack_pressure_level = np.clip((0.0 - slack_mean) / slack_std, 0.0, 2.0)  # 0=loose, ~2=tight
    base_weights = np.array([0.35, 0.25, 0.2, 0.12, 0.08])
    # Boost urgency-sensitive components (latency, energy-gated) under pressure
    weight_scalars = np.array([1.0 + 0.5 * slack_pressure_level,  # latency weight
                               1.0 + 0.3 * slack_pressure_level,  # energy weight (gated)
                               1.0 + 0.2 * slack_pressure_level,  # fairness weight
                               1.0 + 0.4 * slack_pressure_level,  # uncertainty weight
                               1.0])  # work-density weight stays fixed
    dynamic_weights = base_weights * weight_scalars
    dynamic_weights = dynamic_weights / np.sum(dynamic_weights) * 1.0  # renormalize to sum=1.0

    # Base score initialized to neutral
    score = np.full(N, 1.0, dtype=np.float64)

    # Apply hard urgency dominance
    score = np.where(is_urgent, -1000000000000.0, score)

    # Add weighted components only for non-urgent tasks
    score = np.where(is_urgent, score, score + dynamic_weights[0] * norm_critical_latency)
    score = np.where(is_urgent, score, score + dynamic_weights[1] * norm_energy * critical_gate)
    score = np.where(is_urgent, score, score + dynamic_weights[2] * wait_penalty)
    score = np.where(is_urgent, score, score + dynamic_weights[3] * uncertainty_boost)
    score = np.where(is_urgent, score, score + dynamic_weights[4] * work_density_term)

    # Final clipping and NaN protection
    score = np.clip(score, -1000000000000.0, 1000000000000.0)
    score = np.nan_to_num(score, nan=1000000000000.0, posinf=1000000000000.0, neginf=-1000000000000.0)

    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
