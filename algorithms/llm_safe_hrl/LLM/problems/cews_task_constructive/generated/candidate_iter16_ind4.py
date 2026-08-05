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
    v2 mutation: Introduces *slack-activated SEER* (energy/latency ratio gated by slack headroom),
    replaces additive urgency with multiplicative lexicographic gating (urgency >> SEER >> fairness),
    uses robust MAD-based normalization with fallback to zero when variance vanishes,
    replaces sqrt-wait fairness with tanh-scaled aging that saturates smoothly under safety margin,
    introduces *risk-adjusted criticality* = upward_rank * (1 + tanh(uncertainty)) to boost uncertain critical tasks,
    and applies strict zero-energy masking only for negative slack — while preserving monotonic urgency.

    Key innovations:
    - Lexicographic priority via multiplication: urgency_term dominates unless near-zero, then SEER_term activates.
    - SEER = (min_incremental_energy + eps) / (min_exec_time + min_comm_time + eps), gated by exp(-max(0,-slack)/τ) → energy optimization suppressed only *after* violation.
    - Criticality enhanced by uncertainty-aware tanh boost: avoids step-function risk jumps, bounded in [1, 2].
    - Aging term: tanh(ready_wait_time / (1 + abs(slack) + eps)) → saturates at 1.0 under high urgency, stays low when slack < 0.
    - All normalizations use median-absolute-deviation (MAD) with degenerate-safe fallback (no division by zero).
    - Final score = urgency_term * (1 + 0.5 * SEER_term) + risk_criticality_term + aging_term, all finite & deterministic.
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

    def normalize_mad(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size == 1:
            return np.array([0.0])
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        if mad < eps:
            # Degenerate case: all values nearly equal → normalize to zero
            return np.zeros_like(x)
        return (x - med) / (mad + eps)

    # === Urgency term: bounded, monotonic, dominant ===
    # arctan(-slack/tau) ∈ [-π/2, π/2]; scaled to ~[-1.0, 1.0] for clean lex scaling
    tau = 1.0
    urgency_raw = np.arctan(-slack / tau)
    norm_urgency = normalize_mad(urgency_raw)
    # Map to [0.1, 10.0] range to ensure dominance via multiplication later
    urgency_term = np.clip(1.0 + 3.0 * norm_urgency, 0.1, 10.0)

    # === Slack-gated SEER term: energy efficiency only matters *before* violation ===
    # SEER = energy / (exec + comm); higher is worse → we want low SEER for good efficiency
    seer_base = (min_incremental_energy + eps) / (min_exec_time + min_comm_time + eps)
    # Gate: full weight when slack >= 0; exponentially decayed weight when slack < 0 (not zeroed!)
    slack_gate = np.exp(-np.maximum(0.0, -slack) / tau)  # ∈ (0,1] for negative slack; 1.0 for slack ≥ 0
    seer_gated = seer_base * slack_gate
    norm_seer = normalize_mad(seer_gated)
    # Scale to [0.0, 1.0] and invert so lower SEER → higher benefit → lower score contribution
    seer_term = np.clip(1.0 - 0.8 * norm_seer, 0.0, 1.0)

    # === Risk-adjusted criticality: upward_rank boosted by uncertainty, not masked ===
    # tanh(uncertainty) ∈ [0, 1) → multiplier ∈ [1.0, 2.0); preserves ranking order, adds risk-aware emphasis
    risk_boost = 1.0 + np.tanh(uncertainty)
    risk_criticality_base = upward_rank * remaining_work * risk_boost
    norm_criticality = normalize_mad(risk_criticality_base)
    # Invert: higher criticality should yield lower score → prioritize it
    risk_criticality_term = -0.7 * norm_criticality

    # === Fairness (aging): tanh-scaled wait time, saturated under urgency pressure ===
    # When slack is very negative, aging has minimal effect (tanh saturates fast)
    wait_scale = 1.0 + np.abs(slack) + eps
    tanh_aging = np.tanh((ready_wait_time + eps) / wait_scale)
    norm_aging = normalize_mad(tanh_aging)
    # Keep aging modest: contributes only when other terms are balanced
    aging_term = 0.15 * norm_aging

    # === Lexicographic composition: urgency dominates multiplicatively ===
    # Energy-aware boost only applied *when urgency is moderate* (i.e., slack safe) to avoid over-penalizing critical late tasks
    score = urgency_term * (1.0 + 0.5 * seer_term) + risk_criticality_term + aging_term

    # Final sanitization: ensure finite, deterministic, shape-(N,)
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    return score
