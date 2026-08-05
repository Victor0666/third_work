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
    """Novel priority rule emphasizing deadline safety via adaptive risk gating,
    critical-path awareness with workload-normalized urgency, and starvation-aware
    waiting-time boost — all scaled robustly without domain-specific constants.
    
    Key innovations:
      - Deadline risk is gated: only tasks with slack < 0 get strong penalty; 
        others receive mild urgency decay based on relative slack position.
      - Upward rank is weighted by remaining_work to form 'criticality density',
        avoiding blind rank dominance when work is trivial.
      - Energy cost is evaluated *per unit useful work* (incremental_energy / remaining_work),
        promoting energy-efficient progress on high-impact subgraphs.
      - Ready wait time is normalized relative to task's own execution+comm time,
        giving fair starvation relief proportional to expected occupancy.
      - Uncertainty is fused with slack via safe product to penalize risky-late tasks
        more than either alone — avoids additive noise amplification.
      - All features normalized by IQR (robust to outliers) + epsilon, not mean-abs.
      - Final score uses convex combination of deadline-driven and energy-driven terms,
        ensuring DDL feasibility dominates unless all slack >= 0.
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

    # Robust normalization: Interquartile Range (IQR) + eps prevents outlier skew
    def normalize_robust(x):
        q75, q25 = np.percentile(x, [75, 25], method='midpoint')
        iqr = q75 - q25
        return x / (iqr + eps)

    # --- Deadline Risk Component (DDL safety first) ---
    # Strong penalty for negative slack; smooth decay for positive slack
    # Use signed slack-based sigmoid-like shape: steep near zero, flat elsewhere
    slack_norm = normalize_robust(slack)
    deadline_urgency = np.where(
        slack < 0,
        -np.abs(slack) / (np.abs(slack) + eps),  # ≈ -1 when slack << 0, → 0⁻ as slack→0⁻
        np.exp(-slack_norm) - 1.0  # decays from 0 to -0.63 as slack_norm grows; keeps sign
    )

    # --- Critical Path + Work Density ---
    # Upward rank matters more when remaining_work is large → criticality density
    # Avoid division-by-zero: cap remaining_work at eps for safe scaling
    work_safe = np.maximum(remaining_work, eps)
    critical_density = upward_rank * (min_exec_time + min_comm_time) / work_safe
    critical_density = normalize_robust(critical_density)

    # --- Energy Efficiency Signal ---
    # Energy per unit remaining work: lower = more efficient use of energy budget
    energy_per_work = min_incremental_energy / work_safe
    energy_efficiency = normalize_robust(energy_per_work)

    # --- Starvation Relief ---
    # Wait time relative to task's own latency footprint → fair per-task boost
    latency_footprint = np.maximum(min_exec_time + min_comm_time, eps)
    wait_rel = ready_wait_time / latency_footprint
    wait_rel = normalize_robust(wait_rel)

    # --- Risk Amplification ---
    # Multiply uncertainty and deadline risk (safe: both finite, no NaN/inf from product)
    # Only active when slack < 0 or uncertainty > 0; avoids artificial inflation
    risk_amplifier = np.clip(uncertainty * np.abs(deadline_urgency), 0.0, 1e6)
    risk_amplifier = normalize_robust(risk_amplifier)

    # --- Convex Priority Blend ---
    # If ANY task has negative slack → prioritize deadline safety (weight=0.7)
    # Else → shift toward energy efficiency (weight=0.3), but keep criticality & fairness
    has_negative_slack = np.any(slack < 0)
    deadline_weight = 0.7 if has_negative_slack else 0.3
    energy_weight = 1.0 - deadline_weight

    # Base components: all designed so smaller = better (except deadline_urgency: already negative for urgent)
    score_deadline = (
        -deadline_weight * deadline_urgency
        + 0.2 * risk_amplifier
        + 0.1 * critical_density
    )
    
    score_energy = (
        +energy_weight * energy_efficiency
        + 0.15 * critical_density  # still value critical path even in energy mode
        - 0.1 * wait_rel  # higher wait_rel lowers score → boosts priority
    )

    score = score_deadline + score_energy

    # Final safeguard: ensure finite, deterministic output
    return np.nan_to_num(
        score,
        nan=1e12,
        posinf=1e12,
        neginf=-1e12
    )
