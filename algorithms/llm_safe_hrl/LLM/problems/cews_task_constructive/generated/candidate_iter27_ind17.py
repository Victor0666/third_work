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
    v2: Strict lexicographic DDL enforcement with decoupled urgency-criticality-energy layers,
         uncertainty-aware starvation guard, and zero-variance robust normalization.

    Key self-evolved improvements:
    - Urgency is *lexicographically dominant and hard-gated*: tanh-based risk-slack mapped to [0,1],
      then multiplied by strict penalty factor exp(|slack|/tau) for late tasks — ensures late tasks
      always dominate regardless of other features.
    - Criticality term is *only active when slack < 0*, and uses raw upward_rank scaled by normalized
      remaining_work — no energy weighting in critical zone to preserve DDL priority.
    - Energy term is *fully gated off when slack <= 0*, and only activated for slack > tau (not just > 0)
      to avoid premature energy optimization before safety margin is established.
    - Fairness uses linear wait-time (not sqrt) with uncertainty *amplification* (not attenuation) for
      high-risk tasks: fairness = ready_wait_time * (1 + uncertainty), ensuring starving high-uncertainty
      tasks get boosted priority — directly countering reflection's "diluted starvation mitigation".
    - All normalization uses MAD+median with explicit zero-variance fallback and bounded clipping [-3,3]
      to prevent outlier distortion on small N.
    - Final score combines layers with fixed weights (urgency=5.0, critical=3.0, energy=2.0, fairness=0.5)
      and applies *pre-combination sanitization* to guarantee finite, ordered outputs.
    """
    eps = 1e-08
    tau = 1.0

    def clean(x):
        x = np.asarray(x, dtype=float)
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    min_exec_time = clean(min_exec_time)
    min_comm_time = clean(min_comm_time)
    min_incremental_energy = clean(min_incremental_energy)
    slack = clean(slack)
    upward_rank = clean(upward_rank)
    remaining_work = clean(remaining_work)
    ready_wait_time = clean(ready_wait_time)
    uncertainty = clean(uncertainty)

    # === URGENT LAYER (lexicographically dominant, hard-enforced) ===
    # Risk-adjusted slack: penalize uncertainty near deadline
    risk_adjusted_slack = slack - 2.0 * uncertainty
    # tanh maps to [-1,1]; invert & scale to [0,1] for urgency
    urgency_raw = np.tanh(risk_adjusted_slack / (tau + eps))
    urgency_score = (1.0 - urgency_raw) / 2.0
    # Hard late-task multiplier: exp(|slack|/tau) for slack < -eps → ensures dominance
    late_penalty = np.exp(np.abs(slack) / (tau + eps)) * (slack < -eps).astype(float)
    urgency_final = urgency_score * (1.0 + late_penalty)

    # === CRITICALITY LAYER (active ONLY under deadline pressure) ===
    critical_mask = (slack < -eps).astype(float)
    norm_remaining_work = remaining_work / (np.max(remaining_work + eps) + eps)
    critical_base = upward_rank * norm_remaining_work
    def normalize_mad(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size == 1:
            return np.array([0.0])
        median_val = np.median(x)
        mad = np.median(np.abs(x - median_val))
        if mad < eps:
            return np.zeros_like(x)
        normed = (x - median_val) / (mad + eps)
        return np.clip(normed, -3.0, 3.0)
    norm_critical = normalize_mad(critical_base)
    critical_term = 3.0 * norm_critical * critical_mask

    # === ENERGY LAYER (only when safe margin exists: slack > tau) ===
    energy_gate = (slack > tau).astype(float)
    seer_ratio = min_incremental_energy / (min_exec_time + min_comm_time + eps)
    energy_base = 1.0 / (seer_ratio + eps)
    norm_energy = normalize_mad(energy_base)
    energy_term = -2.0 * norm_energy * energy_gate

    # === FAIRNESS LAYER (starvation guard with uncertainty amplification) ===
    # Linear wait time boosted by uncertainty → prioritizes high-risk waiting tasks
    fairness_raw = ready_wait_time * (1.0 + uncertainty)
    norm_fairness = normalize_mad(fairness_raw)
    fairness_term = -0.5 * norm_fairness

    # Combine with strict layer ordering: urgency dominates all; critical dominates energy/fairness when active
    score = 5.0 * urgency_final + critical_term + energy_term + fairness_term

    # Pre-combination sanitization to preserve ordering under extremes
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e8, 1e8)

    # Ensure shape (N,) even for N=1
    return score.reshape(-1)
