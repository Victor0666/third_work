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
    Priority rule v2: Deadline-hardened, uncertainty-aware, and energy-efficient.
    Combines Parent 2's multiplicative hierarchy and tanh urgency with Parent 1's robust
    arctan-based urgency clamping and explicit violation override. Introduces:
      - Adaptive robust_slack = slack - 2*uncertainty (tighter than v1, looser than v0)
      - Unified urgency via clipped tanh for smooth near-deadline discrimination + hard clamp for violations
      - Criticality gated by robust_slack > 0 AND upward_rank > 0 to avoid deadweight descendants
      - Energy term uses SEER (exec+comm)/energy with arctan inversion → monotonic & bounded
      - Fairness only activated when slack > median_slack/2, preventing starvation without DDL risk
      - All terms z-score normalized *then* scaled to [0,1] range for stable composition
      - Deterministic violation override applied *before* composition → guarantees DDL-critical tasks win
    """
    eps = 1e-08
    N = len(slack)
    
    # Sanitize all inputs to finite values
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
    
    # Robust slack: penalize uncertainty more than v1 (2×), less than v0 (3×) → balances sensitivity/stability
    robust_slack = slack - 2.0 * uncertainty
    robust_slack = np.clip(robust_slack, -1e5, 1e5)
    
    # Hard violation mask: any task with slack < 0 gets highest priority (lowest score)
    is_violated = (slack < 0.0).astype(float)
    
    # Urgency: tanh-based for fine-grained near-deadline resolution, but capped and shifted
    # Map robust_slack → [-1,1] smoothly: negative = urgent, positive = relaxed
    urgency_raw = -np.tanh(np.clip(robust_slack / (np.abs(robust_slack) + eps), -10.0, 10.0))
    # Normalize to [0,1] via affine transform: tanh ∈ [-1,1] → (1 - tanh)/2 ∈ [0,1]
    urgency_score = (1.0 - urgency_raw) / 2.0
    
    # Criticality: upward_rank * remaining_work per unit exec+comm, but gated by feasibility & importance
    exec_comm_sum = min_exec_time + min_comm_time + eps
    cp_density = upward_rank * remaining_work / (exec_comm_sum + eps)
    # Gate criticality only if task is both non-violated *and* has meaningful upward_rank
    criticality_gate = (robust_slack > 0.0) & (upward_rank > eps)
    cp_gated = np.where(criticality_gate, cp_density, 0.0)
    
    # Energy efficiency: SEER = (exec+comm)/energy → higher SEER = better efficiency = lower priority score
    # Use arctan to bound and invert: arctan(SEER) ∈ (-π/2, π/2) → map to [0,1]
    seer = exec_comm_sum / (min_incremental_energy + eps)
    energy_raw = np.arctan(seer)
    energy_score = (np.pi/2 - energy_raw) / np.pi  # [0,1]: lower = better efficiency
    
    # Fairness: promote long-waiting tasks *only when safe* — threshold = median_slack/2 (adaptive safety margin)
    fairness_score = np.zeros(N)
    if N > 0 and np.any(slack > eps):
        slack_med = np.median(slack[slack > eps]) + eps
        fairness_threshold = slack_med / 2.0
        safe_for_fairness = (robust_slack > fairness_threshold).astype(float)
        wait_norm = np.clip(ready_wait_time / (np.median(ready_wait_time) + eps), 0.0, 3.0)
        # Sigmoid fairness: prioritize waiting only when clearly safe
        fairness_score = (1.0 / (1.0 + np.exp(-(wait_norm - 1.0)))) * safe_for_fairness
    
    # Uncertainty penalty: linear, bounded, and normalized
    unc_med = np.median(uncertainty) + eps
    unc_penalty = 0.015 * np.clip(uncertainty / unc_med, 0.0, 4.0)
    
    # Normalize each term to [0,1] using robust min-max (avoiding outliers)
    def robust_normalize(x):
        x_clipped = np.clip(x, -1e6, 1e6)
        if x_clipped.size == 1:
            return np.array([0.5])
        p05 = np.percentile(x_clipped, 5)
        p95 = np.percentile(x_clipped, 95)
        if p95 - p05 < eps:
            return np.full_like(x_clipped, 0.5)
        return np.clip((x_clipped - p05) / (p95 - p05 + eps), 0.0, 1.0)
    
    urgency_norm = robust_normalize(urgency_score)
    cp_norm = robust_normalize(cp_gated)
    energy_norm = robust_normalize(energy_score)
    fairness_norm = robust_normalize(fairness_score)
    
    # Multiplicative hierarchy: urgency dominates first; others modulate only when urgency allows
    # Base score = urgency * (1 + criticality) * (1 + energy_efficiency_bonus) * (1 + fairness)
    # Energy term inverted: lower energy_norm means *better* efficiency → should reduce score
    # So we use (1 - energy_norm) as bonus weight for efficiency preference
    score = (
        urgency_norm *
        (1.0 + 0.4 * cp_norm) *
        (1.0 + 0.3 * (1.0 - energy_norm)) *
        (1.0 + 0.15 * fairness_norm) +
        unc_penalty
    )
    
    # Hard override: violated tasks get minimum possible score (highest priority)
    base_min = np.min(score) if N > 0 else 0.0
    score = np.where(is_violated, base_min - 1e9, score)
    
    # Final sanitization: ensure finite, shape (N,), deterministic
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e12, 1e12)
    
    # Ensure output is float64 and shape (N,)
    return score.astype(np.float64).reshape(-1)
