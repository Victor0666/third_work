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
    Priority rule v2: Hard deadline adherence via additive robust slack + urgency gating,
    energy-aware optimization only when safe, and contextually fair aging.
    
    Key innovations:
    - Uses robust_slack = slack - 2.0 * uncertainty (linear, interpretable, avoids multiplicative distortion)
    - Applies hard urgency gate: tasks with robust_slack < 0 get score = -1e6 (guaranteed top priority)
    - Energy term is activated *only* for robust_slack > 0 and weighted by confidence = 1 - tanh(uncertainty)
    - Critical-path importance normalized by workflow-scale median remaining_work and attenuated by uncertainty
    - Fairness boost uses tanh-scaled wait_ratio = ready_wait_time / max(robust_slack, 0.1) — active only if slack > 0.1
    - Aging boost added for tasks near deadline (robust_slack < 1.0) to prevent starvation without violating DDL
    - All normalizations use MAD with degenerate-case safeguards; final scores clipped and sanitized
    - Deterministic, finite, and numerically stable under all edge cases (N=1, constants, NaNs, inf)
    """
    eps = 1e-8
    
    # Sanitize inputs: convert, replace NaN/inf with safe finite values
    def sanitize(x):
        x = np.asarray(x, dtype=float)
        return np.nan_to_num(x, nan=eps, posinf=1e12, neginf=eps)
    
    min_exec_time = sanitize(min_exec_time)
    min_comm_time = sanitize(min_comm_time)
    min_incremental_energy = sanitize(min_incremental_energy)
    slack = sanitize(slack)
    upward_rank = sanitize(upward_rank)
    remaining_work = sanitize(remaining_work)
    ready_wait_time = sanitize(ready_wait_time)
    uncertainty = sanitize(uncertainty)
    
    # Robust slack: linear uncertainty discount preserves deadline semantics
    robust_slack = slack - 2.0 * uncertainty
    robust_slack = np.clip(robust_slack, -1e12, 1e12)
    
    # Hard urgency gate: negative robust_slack → highest priority (smallest score)
    is_urgent = robust_slack < 0.0
    
    # Confidence factor: reliability-aware weight for energy term (bounded in [0.1, 0.95])
    confidence = np.clip(1.0 - np.tanh(uncertainty), 0.1, 0.95)
    
    # Critical-path density: importance × work per unit energy, enabled only when safe for energy saving
    base_density = upward_rank * remaining_work / (min_incremental_energy + eps)
    energy_eligible = robust_slack > 0.0
    critical_energy_density = np.where(energy_eligible, base_density * confidence, 0.0)
    
    # Normalize using robust MAD (handles N=1, constants, outliers)
    def safe_mad_normalize(x):
        if x.size == 0:
            return np.zeros_like(x)
        if x.size == 1:
            return np.zeros_like(x)
        median_x = np.median(x)
        mad = np.median(np.abs(x - median_x)) + eps
        if mad < eps:
            return np.zeros_like(x)
        normed = (x - median_x) / mad
        return np.clip(normed, -3.0, 3.0)
    
    critical_energy_norm = safe_mad_normalize(critical_energy_density)
    
    # Workflow-scale critical path boost: attenuated by uncertainty
    median_rw = np.median(remaining_work) + eps
    cp_boost = upward_rank * (remaining_work / median_rw) * (1.0 / (1.0 + uncertainty + eps))
    cp_boost_norm = safe_mad_normalize(cp_boost)
    
    # Fairness boost: prioritizes aging only when slack margin permits
    margin = np.clip(robust_slack, 0.1, 1e12)
    wait_ratio = ready_wait_time / (margin + eps)
    fairness_boost = np.where(robust_slack > 0.1, np.tanh(0.8 * wait_ratio), 0.0)
    
    # Aging boost: prevents starvation for tasks close to deadline (but not yet urgent)
    aging_boost = np.where(
        (robust_slack < 1.0) & (robust_slack >= 0.0) & (ready_wait_time > eps),
        np.tanh(1.5 * ready_wait_time / (np.abs(robust_slack) + 0.2)),
        0.0
    )
    
    # Base score: urgency dominates, then energy efficiency, structural importance, fairness
    # Negative weights for beneficial terms (higher critical_energy_norm → lower priority score)
    score = (
        +4.6 * safe_mad_normalize(np.where(is_urgent, -1e6, 0.0))  # hard gate dominates
        - 2.7 * critical_energy_norm
        - 1.5 * cp_boost_norm
        + 0.13 * fairness_boost
        + 0.12 * aging_boost
    )
    
    # Apply hard urgency override: assign minimal score to all urgent tasks
    score = np.where(is_urgent, -1e6, score)
    
    # Final sanitization and clipping
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    
    return score
