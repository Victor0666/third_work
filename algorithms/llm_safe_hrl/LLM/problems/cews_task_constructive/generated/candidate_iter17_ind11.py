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
    v2: Hybrid priority rule combining robustness of Parent 2 with critical-path density (CPD)
    from Parent 1, while enforcing strict deadline-feasibility gating and numerical stability.
    
    Key innovations:
    - Uses arctan urgency (Parent 2) for bounded, monotonic, zero-stable deadline sensitivity
    - Integrates CPD = upward_rank / (min_exec_time + eps) as latency-criticality coupling term,
      gated by slack >= 0 to avoid amplifying urgency on violated tasks
    - Replaces latency penalty with *critical-path-aware execution pressure*: 
      (min_exec_time + min_comm_time) * (1 + uncertainty) scaled by urgency magnitude
    - Fairness term uses sqrt(wait) / (1 + |slack| + eps) but capped at 0.25 and activated only when slack > τ_safe
    - All energy-aware terms (CED, CPD) zero-masked when slack < 0 — energy minimization disabled under violation risk
    - Normalization uses adaptive IQR-or-minmax fallback with explicit flat-array safety
    - Final score is sum of normalized, weighted terms; smaller = higher priority
    """
    eps = 1e-08
    tau_safe = 0.1  # minimum slack threshold to activate fairness & uncertainty scaling
    
    # Convert inputs to float arrays, safe against NaN/inf
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    
    def normalize_adaptive(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size == 1:
            return np.array([0.0])
        q25, q75 = np.percentile(x, [25, 75])
        iqr = q75 - q25
        if iqr > eps:
            center = np.median(x)
            scale = iqr + eps
        else:
            xmin, xmax = np.min(x), np.max(x)
            scale = xmax - xmin
            center = (xmax + xmin) / 2.0
            if scale < eps:
                scale = eps
        return (x - center) / (scale + eps)
    
    # 1. Urgency: arctan(-slack/tau) — smooth, bounded [-π/2, π/2], stable near zero
    tau_urgency = 1.0
    urgency_raw = np.arctan(-slack / tau_urgency)
    norm_urgency = normalize_adaptive(urgency_raw)
    urgency_term = -2.6 * norm_urgency  # dominant driver for deadline adherence
    
    # 2. Critical-Energy Density (CED): energy efficiency only matters when feasible
    ced_base = upward_rank * remaining_work / (min_incremental_energy + eps)
    ced_masked = np.where(slack >= 0, ced_base, 0.0)
    norm_ced = normalize_adaptive(ced_masked)
    ced_term = -1.0 * norm_ced  # promotes low-energy, high-importance tasks under feasibility
    
    # 3. Critical-Path Density (CPD): latency-criticality coupling, also feasibility-gated
    cpd_base = upward_rank / (min_exec_time + eps)
    cpd_masked = np.where(slack >= 0, cpd_base, 0.0)
    norm_cpd = normalize_adaptive(cpd_masked)
    cpd_term = -1.4 * norm_cpd  # prioritizes fast execution of high-rank tasks when slack permits
    
    # 4. Execution Pressure: (exec+comm) * (1+uncertainty) scaled by |urgency| — tighter coupling to risk
    base_pressure = min_exec_time + min_comm_time + eps
    uncertainty_factor = 1.0 + uncertainty
    pressure_raw = base_pressure * uncertainty_factor
    # Scale pressure by urgency magnitude only where slack > 0 (avoid over-penalizing violated tasks)
    pressure_scaled = np.where(slack > 0, pressure_raw * (1.0 + np.abs(urgency_raw) / (np.pi/2)), pressure_raw)
    norm_pressure = normalize_adaptive(pressure_scaled)
    pressure_term = 0.85 * norm_pressure  # penalizes slow, uncertain tasks under deadline pressure
    
    # 5. Risk-Tempered Fairness: activated only when slack > tau_safe, avoids starving long-waiting tasks
    wait_safe = np.maximum(ready_wait_time, 0.0)
    sqrt_wait = np.sqrt(wait_safe + eps)
    fairness_scale = 1.0 + np.abs(slack) + eps
    fairness_raw = sqrt_wait / fairness_scale
    fairness_masked = np.where(slack > tau_safe, fairness_raw, 0.0)
    fairness_clipped = np.clip(fairness_masked, 0.0, 0.25)
    norm_fairness = normalize_adaptive(fairness_clipped)
    fairness_term = -0.2 * norm_fairness  # mild boost for aging tasks when deadlines are safe
    
    # 6. Uncertainty penalty only for slack > 0 — separates risk management from violation handling
    uncert_penalty = np.where(slack > 0, uncertainty, 0.0)
    norm_uncert = normalize_adaptive(uncert_penalty)
    uncert_term = 0.3 * norm_uncert
    
    # Aggregate all terms
    score = (
        urgency_term +
        ced_term +
        cpd_term +
        pressure_term +
        fairness_term +
        uncert_term
    )
    
    # Final sanitization: ensure finite, deterministic output
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    
    return score
