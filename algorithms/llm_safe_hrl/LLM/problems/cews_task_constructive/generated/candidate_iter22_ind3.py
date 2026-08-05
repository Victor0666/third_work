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
    v2: Mutated priority rule emphasizing *risk-gated criticality* and *energy-latency tradeoff*
    under deadline hardness. Replaces z-score normalization with MAD-based robust scaling,
    introduces physics-aligned urgency via arctan-scaled slack, replaces multiplicative fairness
    with additive starvation guard only for non-critical tasks, and uses uncertainty-weighted
    critical path density to avoid over-prioritizing high-risk low-work tasks.
    
    Key mutations:
    - Urgency: arctan-based smooth urgency curve with violation boost (no tanh overflow risk)
    - Criticality: uncertainty-weighted CP density (upward_rank * remaining_work / (exec+comm+uncertainty))
    - Energy efficiency: SEER-like ratio normalized via MAD, clipped to [0.01, 100] to prevent inversion
    - Fairness: additive aging term *only* when slack > 3*uncertainty (prevents interference in DDL crisis)
    - Normalization: robust MAD scaling with epsilon fallback for N=1 and outliers
    - Hard violation handling: direct score floor (not relative offset) for slack < -eps
    - Final composition: additive (not multiplicative) to ensure linear dominance hierarchy:
      urgency dominates → criticality → energy → fairness (lexicographic via coefficient weights)
    """
    eps = 1e-8

    def sanitize(x):
        x = np.asarray(x, dtype=float)
        return np.nan_to_num(x, nan=eps, posinf=1e6, neginf=eps)

    min_exec_time = sanitize(min_exec_time)
    min_comm_time = sanitize(min_comm_time)
    min_incremental_energy = sanitize(min_incremental_energy)
    slack = sanitize(slack)
    upward_rank = sanitize(upward_rank)
    remaining_work = sanitize(remaining_work)
    ready_wait_time = sanitize(ready_wait_time)
    uncertainty = sanitize(uncertainty)

    # Robust urgency: arctan maps slack ∈ ℝ → (-π/2, π/2), then shift/scale to [0, 1]
    # Negative slack yields strong urgency; zero slack → ~0.5; large positive → near 0
    urgency_raw = np.arctan(-slack / (uncertainty + eps)) / np.pi + 0.5
    # Boost urgency for violated deadlines (slack < -eps) → set to max urgency = 1.0
    urgency_raw = np.where(slack < -eps, 1.0, urgency_raw)

    # Criticality: uncertainty-gated CP density — penalize high-uncertainty low-density tasks
    exec_comm_unc = min_exec_time + min_comm_time + uncertainty + eps
    cp_density = (upward_rank * remaining_work) / exec_comm_unc
    # Gate criticality by positive slack to avoid over-scheduling risky tasks when deadline is tight
    cp_gate = np.where(slack > 2.0 * uncertainty, 1.0, 0.3)  # soft gate, not hard zero
    cp_density_gated = cp_density * cp_gate

    # Energy efficiency: inverse SEER (energy per latency unit), robustly scaled
    seer_raw = min_incremental_energy / (min_exec_time + min_comm_time + eps)
    # Clip extreme values before normalization to preserve ordinal relation
    seer_clipped = np.clip(seer_raw, 0.01, 100.0)

    # Fairness: additive aging term *only* when slack allows (no starvation in crisis mode)
    fairness_raw = np.where(slack > 3.0 * uncertainty, ready_wait_time / (slack + eps), 0.0)

    # Robust MAD-based normalization (more outlier-resistant than std)
    def normalize_mad(x):
        x = np.clip(x, -1e6, 1e6)
        if x.size == 1:
            return np.array([0.0])
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        if mad < eps:
            return np.zeros_like(x)
        return (x - med) / (mad + eps)

    norm_urgency = normalize_mad(urgency_raw)
    norm_cp = normalize_mad(cp_density_gated)
    norm_seer = normalize_mad(seer_clipped)
    norm_fair = normalize_mad(fairness_raw)

    # Lexicographic weighting: urgency dominates (weight 10), then CP (3), energy (2), fairness (1)
    # All terms shifted to [0, ∞) to avoid negative scores that invert priority meaning
    score_urgency = 10.0 * np.clip(norm_urgency, 0.0, np.inf)
    score_cp = 3.0 * np.clip(norm_cp, 0.0, np.inf)
    score_energy = 2.0 * np.clip(norm_seer, 0.0, np.inf)
    score_fairness = 1.0 * np.clip(norm_fair, 0.0, np.inf)

    # Base score: sum of weighted components
    score = score_urgency + score_cp + score_energy + score_fairness

    # Hard violation override: assign minimum possible score (highest priority) for slack < -eps
    violation_mask = (slack < -eps).astype(float)
    min_score = np.min(score) if len(score) > 0 else 0.0
    score = np.where(violation_mask, min_score - 1e9, score)

    # Final sanitization: ensure finite, bounded output
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e12, 1e12)

    return score
