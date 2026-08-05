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
    v2 mutation: Replaces additive composition with lexicographic multiplicative priority hierarchy.
    Introduces:
      - Urgency: clipped tanh(-slack/τ) for strict hard-deadline dominance (τ=0.5s)
      - Critical-path density (CPD): upward_rank / (min_exec_time + ε), scaled by slack safety
      - SEER-gated energy efficiency: (min_exec_time + min_comm_time) / (min_incremental_energy + ε),
        activated only when slack > 2.0s to avoid diluting urgency
      - Risk-adjusted fairness: sqrt(ready_wait_time + ε) * exp(-uncertainty), capped and slack-gated
      - All terms normalized via robust MAD-based scaling to [-2, 2], preserving gradient & stability
      - Final score = urgency × (1 + CPD) × (1 + SEER_term) × (1 + fairness_term),
        ensuring multiplicative priority preservation and zero-energy masking for violated deadlines
    """
    eps = 1e-8
    # Ensure float arrays, no in-place modification
    min_exec_time = np.asarray(min_exec_time, dtype=float).copy()
    min_comm_time = np.asarray(min_comm_time, dtype=float).copy()
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float).copy()
    slack = np.asarray(slack, dtype=float).copy()
    upward_rank = np.asarray(upward_rank, dtype=float).copy()
    remaining_work = np.asarray(remaining_work, dtype=float).copy()
    ready_wait_time = np.asarray(ready_wait_time, dtype=float).copy()
    uncertainty = np.asarray(uncertainty, dtype=float).copy()

    def robust_normalize(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size == 1:
            return np.array([0.0])
        # Use MAD (median absolute deviation) for robust scaling
        med = np.median(x)
        dev = np.abs(x - med)
        mad = np.median(dev)
        scale = mad if mad > eps else (np.max(x) - np.min(x) + eps)
        # Clamp to [-2, 2] after normalization
        z = (x - med) / (scale + eps)
        return np.clip(z, -2.0, 2.0)

    # === URGENCY TERM: Hard deadline dominance via bounded tanh ===
    # Stronger penalty for negative slack; smooth transition near zero
    tau_urgency = 0.5
    urgency_raw = np.tanh(-slack / tau_urgency)  # [-1, 1]; negative slack → +1 (high priority)
    norm_urgency = robust_normalize(urgency_raw)
    # Map to [0.1, 10.0] multiplicative base: higher urgency → smaller multiplier → higher priority
    urgency_mult = np.clip(0.1 + 9.9 * (1.0 - (norm_urgency + 2.0) / 4.0), 0.1, 10.0)

    # === CRITICAL-PATH DENSITY (CPD): Latency-sensitive criticality ===
    # Prioritize tasks with high upward_rank but low exec time — "bottleneck accelerators"
    cpd_base = upward_rank / (min_exec_time + eps)
    # Gate CPD: reduce weight when slack is dangerously low (< 0.1s) to avoid over-accelerating non-urgent paths
    cpd_gate = np.where(slack > 0.1, 1.0, 0.3)
    cpd_scaled = cpd_base * cpd_gate
    norm_cpd = robust_normalize(cpd_scaled)
    # Map to [0.0, 1.0] additive offset → becomes multiplicative factor (1 + offset)
    cpd_offset = np.clip((norm_cpd + 2.0) / 4.0, 0.0, 1.0)

    # === SEER-GATED ENERGY EFFICIENCY (SEER = latency/energy) ===
    # Only considered when slack > 2.0s (safe margin); otherwise ignored (no energy trade-off under risk)
    seer_base = (min_exec_time + min_comm_time + eps) / (min_incremental_energy + eps)
    seer_active = np.where(slack > 2.0, seer_base, 0.0)
    norm_seer = robust_normalize(seer_active)
    # Map to [0.0, 0.5] to avoid overwhelming urgency; higher SEER → better efficiency → lower priority cost
    seer_offset = np.clip((2.0 - norm_seer) / 4.0, 0.0, 0.5)  # inverted: better SEER → larger offset

    # === RISK-ADJUSTED FAIRNESS ===
    # Boost long-waiting tasks, but dampen under high uncertainty or tight slack
    wait_boost = np.sqrt(np.maximum(ready_wait_time, 0.0) + eps)
    uncertainty_damp = np.exp(-np.clip(uncertainty, 0.0, 10.0))
    fairness_raw = wait_boost * uncertainty_damp
    # Gate fairness: only activate when slack > 1.0s (avoid starving urgent tasks)
    fairness_gated = np.where(slack > 1.0, fairness_raw, 0.0)
    norm_fairness = robust_normalize(fairness_gated)
    # Map to [0.0, 0.3] additive offset
    fairness_offset = np.clip((norm_fairness + 2.0) / 4.0 * 0.3, 0.0, 0.3)

    # === MULTIPLICATIVE COMPOSITION ===
    # Lexicographic priority: urgency dominates; CPD adds critical-path sensitivity;
    # SEER and fairness provide secondary refinement *only* in safe regions
    score = (
        urgency_mult *
        (1.0 + cpd_offset) *
        (1.0 + seer_offset) *
        (1.0 + fairness_offset)
    )

    # Ensure finite output and deterministic shape
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=1e9)
    return score
