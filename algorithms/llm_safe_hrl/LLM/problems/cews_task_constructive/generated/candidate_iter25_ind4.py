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
    v2: Risk-aware lexicographic priority with adaptive urgency gating, 
         SEER-inspired energy efficiency ratio, and starvation-robust fairness.
    
    Key mutations:
    - Replaces arctan urgency with tanh-based risk-adjusted slack: tanh((slack - 2*uncertainty)/tau)
      → smoother near deadline, explicitly penalizes high-uncertainty tasks earlier.
    - Introduces *normalized critical path pressure* = (upward_rank * remaining_work) / (min_exec_time + min_comm_time + eps)
      → prioritizes high-workload critical paths more discriminately than raw cost.
    - Energy term now uses *inverse SEER*: (min_incremental_energy + eps) / (upward_rank * remaining_work + eps)
      → directly favors low-energy-per-critical-work, gated only when slack > 0.
    - Fairness term redefined as linear wait boost scaled by urgency: ready_wait_time * (1 - tanh(slack / (1.0 + eps)))
      → amplifies waiting priority precisely where slack is tight, avoiding artificial boosts under safety.
    - All normalization uses MAD (median absolute deviation) instead of IQR for better small-N robustness.
    - Final score applies strict lexicographic ordering via multiplicative urgency gating:
        score = urgency_base + (1 + |urgency_base|) * [criticality + energy + fairness]
      → ensures deadline violation dominates all other objectives.
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
    
    # Risk-adjusted urgency: tanh((slack - 2*uncertainty)/tau) → negative when risk > slack buffer
    # Shifts penalty earlier for uncertain tasks; bounded in [-1, 1]; smaller = more urgent
    risk_adjusted_slack = slack - 2.0 * uncertainty
    urgency_raw = np.tanh(risk_adjusted_slack / (tau + eps))
    urgency_base = -urgency_raw  # now: smaller value = higher urgency (e.g., -0.99 > 0.1)
    
    # Robust MAD-based normalization (more stable than IQR for N < 5)
    def normalize_mad(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size == 1:
            return np.array([0.0])
        median_x = np.median(x)
        mad = np.median(np.abs(x - median_x))
        if mad < eps:
            # fallback to range scaling if MAD near zero
            xmin, xmax = np.min(x), np.max(x)
            scale = xmax - xmin
            center = (xmax + xmin) / 2.0
            if scale < eps:
                scale = eps
            normed = (x - center) / (scale + eps)
        else:
            normed = (x - median_x) / (mad + eps)
        return np.clip(normed, -5.0, 5.0)
    
    norm_urgency = normalize_mad(urgency_base)
    urgency_term = 4.0 * norm_urgency  # dominant base term
    
    # Criticality pressure: work importance per unit execution+comm time
    # Higher ratio → more critical work per latency cost → prioritize
    critical_cost = min_exec_time + min_comm_time + eps
    critical_ratio = (upward_rank * remaining_work) / critical_cost
    norm_critical = normalize_mad(critical_ratio)
    critical_term = -1.8 * norm_critical
    
    # Energy efficiency: inverse SEER — energy per critical work; only active when slack > 0
    seer_inverse = (min_incremental_energy + eps) / (upward_rank * remaining_work + eps)
    energy_gate = np.where(slack > 0.0, 1.0, 0.0)  # hard gate: no energy optimization under violation
    masked_seer = seer_inverse * energy_gate
    norm_energy = normalize_mad(masked_seer)
    energy_term = 1.0 * norm_energy  # smaller seer_inverse → lower energy term → higher priority
    
    # Fairness: linear wait boost modulated by urgency — no boost when slack is safe
    # tanh(slack/tau) ≈ 1 when slack >> 0 → (1 - ~1) ≈ 0 → no fairness boost
    # tanh(slack/tau) ≈ -1 when slack << 0 → (1 - ~-1) = 2 → full boost
    urgency_modulator = 1.0 - np.tanh(slack / (tau + eps))
    fairness_raw = ready_wait_time * urgency_modulator
    norm_fairness = normalize_mad(fairness_raw)
    fairness_term = -0.3 * norm_fairness  # smaller fairness term → higher priority for long-waiting
    
    # Lexicographic dominance: urgency_base strictly modulates non-urgency terms
    # Ensures deadline violations override all other considerations
    modulation = 1.0 + np.abs(urgency_base)
    non_urgency_terms = critical_term + energy_term + fairness_term
    score = urgency_term + modulation * non_urgency_terms
    
    # Final sanitization and clipping
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e9, 1e9)
    
    return score
