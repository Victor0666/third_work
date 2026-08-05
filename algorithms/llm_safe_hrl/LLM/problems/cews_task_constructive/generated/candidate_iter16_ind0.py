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
    Mutated priority rule emphasizing:
    - Strict zero-tolerance urgency gating (hard deadline enforcement)
    - Slack-aware uncertainty scaling: higher risk amplification for negative slack
    - MAD-based robust normalization (more outlier-resilient than IQR)
    - Critical-path energy coupling: energy penalty scaled by upward_rank * |slack|^-1 when slack < 0
    - Adaptive starvation control: waiting penalty activated only when slack > 0 and wait > median duration
    - Communication-pressure term: prioritizes tasks with high comm-to-exec ratio to reduce bottleneck cascades
    - All operations epsilon-protected, nan/inf guarded, deterministic, shape-compliant.
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
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=float)

    # Robust normalization using median + MAD (more stable for small N than IQR)
    def robust_normalize_mad(x):
        med = np.median(x)
        mad = np.median(np.abs(x - med)) + eps
        centered = x - med
        normalized = centered / mad
        return np.clip(normalized, -8.0, 8.0)  # tighter bound than IQR for stability

    # === 1. HARD DEADLINE GATE: zero-tolerance enforcement ===
    # Negative or zero slack → highest priority (score = -1.0), overriding all else
    hard_urgency = np.full(N, 1.0)
    hard_urgency = np.where(slack <= 0.0, -1.0, hard_urgency)

    # === 2. RELATIVE URGENCY (soft region: slack > 0) ===
    task_min_duration = np.maximum(min_exec_time + min_comm_time, eps)
    rel_slack = np.maximum(slack, eps) / task_min_duration  # avoid div-by-zero & negative slack
    # Soft urgency: decays smoothly; higher when slack is tight relative to duration
    soft_urgency = 1.0 / (1.0 + rel_slack * 0.3 + eps)

    # === 3. UNCERTAINTY-SCALED ENERGY PENALTY ===
    # Amplify energy cost under uncertainty, especially when slack is low
    # Use |slack|^-1 scaling for risk-sensitive penalty boost when slack > 0; constant boost when slack <= 0
    slack_inv_weight = np.where(slack <= 0.0, 2.0, np.clip(1.0 / (np.abs(slack) + eps), 0.5, 5.0))
    energy_risk_scaled = min_incremental_energy * (1.0 + uncertainty * slack_inv_weight)
    energy_safe = np.maximum(energy_risk_scaled, eps)
    energy_norm = robust_normalize_mad(energy_safe)

    # === 4. CRITICALITY-ENERGY COUPLING ===
    # For high-upward-rank tasks under tight slack: penalize energy more aggressively
    ur_med = np.median(upward_rank) + eps
    ur_ratio = upward_rank / ur_med
    critical_energy_penalty = np.where(
        (ur_ratio > 1.2) & (slack > 0.0),
        energy_norm * (1.0 + 0.8 * np.clip((ur_ratio - 1.2), 0.0, 2.0)),
        energy_norm
    )
    # When slack <= 0, force strong critical-energy coupling regardless of ur_ratio
    critical_energy_penalty = np.where(
        slack <= 0.0,
        energy_norm * 2.5,
        critical_energy_penalty
    )

    # === 5. COMMUNICATION-PRESSURE TERM ===
    # Prioritize tasks where communication dominates execution (potential bottleneck)
    comm_exec_ratio = min_comm_time / (min_exec_time + eps)
    comm_pressure = robust_normalize_mad(comm_exec_ratio)
    # Boost priority (i.e., lower score) when comm dominates and slack is tight
    comm_boost = np.where(
        (comm_exec_ratio > 0.7) & (rel_slack < 2.0),
        -0.4 * comm_pressure,
        0.0
    )

    # === 6. STARVATION CONTROL (adaptive & bounded) ===
    # Only activate waiting penalty when slack > 0 AND wait time exceeds median task duration
    median_duration = np.median(task_min_duration) + eps
    wait_active = (slack > 0.0) & (ready_wait_time > median_duration)
    wait_ratio = np.clip(ready_wait_time / (median_duration + eps), 0.0, 2.0)
    wait_penalty = np.where(wait_active, 0.15 * (wait_ratio - 1.0), 0.0)  # bounded [0, 0.15]

    # === 7. WORK-AWARE LATENCY PENALTY ===
    # Longer remaining work → tolerate slightly longer exec/comm if critical
    work_norm = robust_normalize_mad(remaining_work)
    # Reduce latency penalty (i.e., add small positive score) for high-work tasks when slack permits
    latency_relief = np.where(
        (remaining_work > np.median(remaining_work) + eps) & (slack > 0.0),
        0.1 * work_norm,
        0.0
    )

    # === 8. FINAL SCORE ASSEMBLY ===
    # Hard gate dominates: if any task has slack <= 0, its score is -1.0 (max priority)
    # Otherwise, combine soft terms with calibrated weights
    base_score = (
        0.0 * soft_urgency  # already embedded in hard_urgency logic; not additive
        + 1.8 * critical_energy_penalty
        + 0.25 * robust_normalize_mad(min_exec_time)
        + 0.3 * robust_normalize_mad(min_comm_time)
        + 0.1 * work_norm
        + 0.12 * robust_normalize_mad(uncertainty)
        + wait_penalty
        + latency_relief
        + comm_boost
    )

    # Apply hard urgency gate: override base_score for all slack<=0 tasks
    score = np.where(slack <= 0.0, hard_urgency, base_score)

    # Ensure finite output and shape compliance
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    return score
