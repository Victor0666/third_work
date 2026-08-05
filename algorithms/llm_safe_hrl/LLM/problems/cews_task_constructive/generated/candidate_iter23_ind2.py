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
    v2 priority rule: Urgency-dominant + latency-gated criticality + risk-normalized energy +
                      sigmoid-modulated fairness + uncertainty-scaled energy + robust local normalization.
    
    Key synthesis:
    - Keeps Parent 2's stable sigmoid-based wait pressure and clean MAD normalization.
    - Adopts Parent 1's *local trimmed norm* fallback for N=1 (zero vector) and explicit eps safety.
    - Uses Parent 2's smooth slack_sigmoid for fairness activation, but adds Parent 1's *sliding slack sensitivity*
      via 1/|slack| clipping to boost urgency resolution near zero.
    - Integrates uncertainty multiplicatively into energy denominator (Parent 2) AND adds dedicated
      uncertainty-boosted slack sensitivity (Parent 1) — but only when slack is tight (rel_slack <= 0.3).
    - Eliminates all dynamic weights and fragile gates; uses fixed hierarchical layering.
    - Introduces *work-density bonus*: negative contribution for high remaining_work when slack > 0,
      encouraging early scheduling of heavy sub-DAGs under safe conditions.
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
        if N == 1:
            return np.zeros_like(x_clean)
        med = np.median(x_clean)
        abs_dev = np.abs(x_clean - med)
        mad = np.median(abs_dev)
        if mad < eps:
            return np.zeros_like(x_clean)
        z = (x_clean - med) / (mad + eps)
        return np.clip(z, -3.0, 3.0)

    # Urgency mask: hard deadline violation → top priority
    is_urgent = (slack <= 0.0).astype(np.float64)

    # Duration and relative slack for gating
    duration = min_exec_time + min_comm_time + eps
    rel_slack = np.divide(slack, duration, out=np.zeros_like(slack), where=duration != 0)
    rel_slack = np.nan_to_num(rel_slack, nan=0.0, posinf=0.0, neginf=0.0)
    tight_slack_mask = (rel_slack <= 0.3).astype(np.float64)

    # Critical latency: duration * upward_rank, robustly normalized
    critical_latency_raw = duration * upward_rank
    norm_critical_latency = robust_mad_norm(critical_latency_raw)

    # Risk-normalized energy: energy / [duration * (1 + uncertainty)]
    effective_duration = duration * (1.0 + uncertainty + eps)
    risk_norm_energy = np.divide(
        min_incremental_energy,
        effective_duration,
        out=np.zeros_like(min_incremental_energy),
        where=effective_duration != 0
    )
    risk_norm_energy = np.nan_to_num(risk_norm_energy, nan=0.0, posinf=0.0, neginf=0.0)
    norm_energy = robust_mad_norm(risk_norm_energy)

    # Sigmoid-driven wait pressure: amplifies fairness when slack is low
    slack_median_dur = np.median(duration) + eps
    slack_sigmoid = 1.0 / (1.0 + np.exp(-slack / slack_median_dur))
    wait_per_duration = np.divide(
        ready_wait_time,
        duration,
        out=np.zeros_like(ready_wait_time),
        where=duration != 0
    )
    wait_per_duration = np.nan_to_num(wait_per_duration, nan=0.0, posinf=0.0, neginf=0.0)
    norm_wait_rel = robust_mad_norm(wait_per_duration)
    wait_pressure = (1.0 - slack_sigmoid) * norm_wait_rel

    # Slack sensitivity boost: 1/|slack| clipped, enhanced by uncertainty when tight
    abs_slack = np.abs(slack) + eps
    slack_sensitivity_raw = np.clip(1.0 / abs_slack, 0.1, 20.0)
    uncertainty_boost = slack_sensitivity_raw * uncertainty * tight_slack_mask

    # Work-density bonus: prioritize large remaining work only when slack > 0 (safe to schedule early)
    safe_to_schedule = (slack > 0.0).astype(np.float64)
    work_density_bonus = -0.07 * robust_mad_norm(remaining_work) * safe_to_schedule

    # Fixed-weight layered score (urgency dominates unconditionally)
    score = np.full(N, 0.0, dtype=np.float64)
    score = np.where(is_urgent, -1000000000000.0, score)
    score = np.where(is_urgent, score, score + 0.42 * norm_critical_latency * tight_slack_mask)
    score = np.where(is_urgent, score, score + 0.26 * norm_energy * tight_slack_mask)
    score = np.where(is_urgent, score, score + 0.18 * wait_pressure)
    score = np.where(is_urgent, score, score + 0.09 * uncertainty_boost)
    score = np.where(is_urgent, score, score + work_density_bonus)

    # Final sanitization
    score = np.clip(score, -1000000000000.0, 1000000000000.0)
    score = np.nan_to_num(score, nan=1000000000000.0, posinf=1000000000000.0, neginf=-1000000000000.0)
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
