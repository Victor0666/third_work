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
    Hybrid priority rule combining robust deadline gating (v0) with smooth urgency decay (v1),
    workload-normalized criticality density, latency-relative aging, and multiplicative risk fusion.
    
    Key improvements:
    - Uses adaptive threshold from v0 *and* smooth slack decay from v1: combines hard safety guard
      with graceful prioritization for near-deadline tasks.
    - Critical density now includes communication time *and* uses min_exec_time+min_comm_time as natural
      latency footprint — improves sensitivity to data-intensive tasks.
    - Aging boost is latency-relative *and* capped to prevent dominance, preserving starvation fairness.
    - Risk amplifier uses clipped product of uncertainty and absolute deadline urgency, then IQR-normalized.
    - Final convex weighting dynamically adjusts deadline/energy emphasis based on worst-case slack,
      not just binary existence of negative slack — enables finer-grained control.
    - All divisions guarded; all outputs finite, deterministic, and shape-(N,) compliant.
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
    
    # Robust IQR normalization helper
    def normalize_iqr(x):
        q75, q25 = np.percentile(x, 75), np.percentile(x, 25)
        iqr = q75 - q25
        scale = iqr if iqr > eps else np.mean(np.abs(x)) + eps
        return x / (scale + eps)
    
    # Adaptive deadline risk: hard gate for severe lateness + smooth decay for marginal slack
    median_slack_pos = np.median(slack[slack > 0]) if np.any(slack > 0) else 1.0
    critical_slack_threshold = np.minimum(-0.1, -0.05 * median_slack_pos)
    slack_norm = normalize_iqr(slack)
    # Hard penalty for severely late tasks; smooth exponential decay for others
    deadline_urgency = np.where(
        slack < critical_slack_threshold,
        -slack / (np.abs(slack) + eps),
        np.exp(-slack_norm) - 1.0
    )
    
    # Latency footprint: execution + communication (natural duration proxy)
    latency_footprint = min_exec_time + min_comm_time
    latency_footprint_safe = np.maximum(latency_footprint, eps)
    
    # Criticality density: upward rank weighted by work-per-latency, normalized
    work_safe = np.maximum(remaining_work, eps)
    critical_density = upward_rank * latency_footprint_safe / work_safe
    critical_density = normalize_iqr(critical_density)
    
    # Energy efficiency per useful work unit (joules per MI-equivalent duration)
    energy_per_work = min_incremental_energy / work_safe
    energy_efficiency = normalize_iqr(energy_per_work)
    
    # Fair aging: relative wait time, capped to avoid overwhelming other signals
    wait_rel = np.clip(ready_wait_time / latency_footprint_safe, 0.0, 5.0)
    wait_rel = normalize_iqr(wait_rel)
    
    # Risk amplifier: multiplicative fusion of uncertainty and deadline urgency, robustly normalized
    risk_base = uncertainty * np.abs(deadline_urgency)
    risk_amplifier = np.clip(risk_base, 0.0, 1e6)
    risk_amplifier = normalize_iqr(risk_amplifier)
    
    # Dynamic deadline weight: increases with worst slack violation severity (not just presence)
    worst_slack = np.min(slack)
    # Weight ramps from 0.3 (all slack >= 0) to 0.9 (severe violation), smoothly via sigmoid-like scaling
    deadline_weight = 0.3 + 0.6 * (1.0 / (1.0 + np.exp(-(worst_slack - critical_slack_threshold) / 0.5)))
    energy_weight = 1.0 - deadline_weight
    
    # Deadline-driven score: urgency dominates, amplified by risk and criticality
    score_deadline = -deadline_weight * deadline_urgency + 0.25 * risk_amplifier + 0.15 * critical_density
    
    # Energy-driven score: efficiency promoted, moderated by criticality and starvation relief
    score_energy = energy_weight * energy_efficiency + 0.1 * critical_density - 0.15 * wait_rel
    
    # Final score: sum, then sanitize
    score = score_deadline + score_energy
    return np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
