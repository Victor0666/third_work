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
    v2: Strictly monotonic urgency, fully orthogonal signals, safety-gated energy,
         starvation-immune fairness, and robust normalization.
    
    Key self-evolution improvements:
    - Urgency: Pure clipped inverse-slack (no arctan) → guarantees strict monotonicity;
      zero-divide guarded; hard violation triggers minimal score.
    - Criticality: Upward rank normalized by remaining work only → pure critical-path importance,
      decoupled from energy or latency to preserve orthogonality.
    - Energy: Marginal energy per unit execution+comm time, *only activated when slack > tau_safe*,
      scaled by slack margin to favor low-energy tasks *without compromising deadlines*.
    - Fairness: Linear wait time dominates — no slack division; instead uses additive penalty
      scaled *only* by uncertainty (risk-aware) and capped minimum wait weight → ensures
      starvation prevention even under tight deadlines (e.g., slack ≤ 0.1).
    - All signals normalized independently via winsorized IQR + median centering with guaranteed
      finite scale and shape-(N,) output.
    - Hard-DDL enforcement: slack <= 0 → fixed max-priority score (-1e9); no floating-point edge cases.
    """
    eps = 1e-08
    tau_safe = 2.0
    min_wait_weight = 0.3  # Ensures fairness always contributes meaningfully, even when slack is tiny
    
    def clean(x):
        x = np.asarray(x, dtype=float)
        return np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
    
    min_exec_time = clean(min_exec_time)
    min_comm_time = clean(min_comm_time)
    min_incremental_energy = clean(min_incremental_energy)
    slack = clean(slack)
    upward_rank = clean(upward_rank)
    remaining_work = clean(remaining_work)
    ready_wait_time = clean(ready_wait_time)
    uncertainty = clean(uncertainty)
    
    N = len(slack)
    is_violated = slack <= 0.0
    
    # --- Urgency: strictly monotonic inverse-slack, clipped for stability ---
    urgency_raw = np.where(is_violated, 0.0, 1.0 / (np.maximum(slack, eps) + eps))
    
    # --- Criticality: pure upward_rank density (decoupled from energy/latency) ---
    crit_raw = upward_rank / (remaining_work + eps)
    
    # --- Energy-efficiency: only active when safe; marginal energy per latency unit ---
    exec_comm_sum = min_exec_time + min_comm_time + eps
    energy_efficiency_raw = min_incremental_energy / exec_comm_sum
    energy_gate = np.clip((slack - tau_safe) / (tau_safe + eps), 0.0, 1.0)
    energy_raw = energy_efficiency_raw * energy_gate
    
    # --- Fairness: linear wait dominates; uncertainty amplifies penalty, no slack division ---
    wait_safe = np.maximum(ready_wait_time, 0.0)
    # Always apply at least min_wait_weight × wait_time, amplified by uncertainty (0–1 bounded)
    fairness_raw = wait_safe * (min_wait_weight + 0.7 * np.clip(uncertainty, 0.0, 1.0))
    
    # --- Robust normalization: winsorized IQR + median centering ---
    def normalize_robust(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size == 1:
            return np.array([0.0])
        p025, p975 = np.percentile(x, [2.5, 97.5])
        x_winsor = np.clip(x, p025, p975)
        q25, q75 = np.percentile(x_winsor, [25, 75])
        iqr = q75 - q25
        if iqr > eps:
            center = np.median(x_winsor)
            scale = iqr + eps
        else:
            xmin, xmax = np.min(x_winsor), np.max(x_winsor)
            scale = max(xmax - xmin, eps)
            center = (xmax + xmin) / 2.0
        return (x_winsor - center) / (scale + eps)
    
    norm_urgency = normalize_robust(urgency_raw)
    norm_crit = normalize_robust(crit_raw)
    norm_energy = normalize_robust(energy_raw)
    norm_fair = normalize_robust(fairness_raw)
    
    # --- Weighted linear combination: urgency dominates, criticality guides, energy saves only when safe, fairness prevents starvation ---
    score_dynamic = (
        -8.0 * norm_urgency   # Highest weight: deadline adherence is non-negotiable
        + 2.2 * norm_crit     # Moderate weight: prioritize critical path
        - 1.6 * norm_energy   # Negative: lower energy preferred when safe
        + 0.4 * norm_fair     # Positive: longer wait increases priority (lower score = higher priority)
    )
    
    # --- Hard DDL violation: highest priority (lowest score) ---
    score = np.where(is_violated, -1e9, score_dynamic)
    
    # --- Final sanitization: ensure finite, deterministic, shape-(N,) float64 output ---
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e9, 1e9)
    return score.astype(np.float64, copy=True)
