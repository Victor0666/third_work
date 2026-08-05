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
    v2: Deadline-strict, criticality-prioritized, energy-aware, and numerically robust priority.
    
    Key improvements:
    - Uses arctan urgency (bounded, monotonic, sharper near deadline) from Parent 2
    - Robust IQR/median normalization (Parent 2) with explicit N=1 handling and clipping
    - Criticality-pressure term activated early (slack > -1.0s) with multiplicative uncertainty
    - Energy-awareness gated smoothly via sigmoid (Parent 2), but strengthened by SEER-inspired
      work-criticality ratio (upward_rank * remaining_work / energy) for better energy-efficiency tradeoff
    - Fairness term simplified to sqrt(wait) / (1 + max(0, slack) + eps) — avoids penalizing waiting
      under violation while preserving fairness when deadlines are safe
    - Uncertainty penalty removed (redundant with pressure term) and replaced by direct uncertainty
      scaling in criticality-pressure for tighter coupling
    - All divisions guarded; NaN/inf explicitly sanitized before normalization; final score bounded
    """
    eps = 1e-08
    tau_violation = 1.0

    # Sanitize inputs: convert to float, replace NaN/inf with safe values
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

    # Urgency: arctan(-slack) — monotonic, bounded [-pi/2, pi/2], sharp near zero slack
    urgency_raw = np.arctan(-slack / (1.0 + eps))
    
    # Normalize robustly using IQR/median; handles N=1 safely
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
        normed = (x - center) / (scale + eps)
        return np.clip(normed, -5.0, 5.0)
    
    norm_urgency = normalize_adaptive(urgency_raw)
    urgency_term = -3.2 * norm_urgency  # Slightly stronger urgency weight

    # Criticality-energy density (CED): upward_rank * remaining_work / energy, gated by slack safety
    ced_base = upward_rank * remaining_work / (min_incremental_energy + eps)
    # Smooth sigmoid gate: active when slack is non-negative, ramps up gradually
    gate_sigmoid = 1.0 / (1.0 + np.exp(-(slack + 0.5 * tau_violation) / (0.25 * tau_violation + eps)))
    ced_masked = ced_base * gate_sigmoid
    norm_ced = normalize_adaptive(ced_masked)
    ced_term = -1.6 * norm_ced  # Increased weight for energy efficiency under safety

    # Criticality-pressure: upward_rank * (exec+comm) * (1+uncertainty), activated for slack > -1.0s
    critical_cost = min_exec_time + min_comm_time + eps
    pressure_base = upward_rank * critical_cost * (1.0 + uncertainty)
    # Linear ramp gate from slack = -tau_violation to 0 → full activation at slack >= 0
    pressure_gate = np.clip((slack + tau_violation) / (tau_violation + eps), 0.0, 1.0)
    pressure_masked = pressure_base * pressure_gate
    norm_pressure = normalize_adaptive(pressure_masked)
    pressure_term = 1.2 * norm_pressure  # Slightly reduced to balance CED emphasis

    # Fairness: reward long-waiting tasks only when deadlines are safe (slack >= 0)
    wait_safe = np.maximum(ready_wait_time, 0.0)
    sqrt_wait = np.sqrt(wait_safe + eps)
    # Fairness scale: grows with slack safety; no penalty when violating
    fairness_scale = 1.0 + np.maximum(slack, 0.0) + eps
    fairness_raw = sqrt_wait / fairness_scale
    norm_fairness = normalize_adaptive(fairness_raw)
    fairness_term = -0.25 * norm_fairness  # Reduced weight for cleaner prioritization

    # Assemble final score: smaller = higher priority
    score = urgency_term + ced_term + pressure_term + fairness_term

    # Final sanitization: ensure finite, bounded output
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e9, 1e9)

    return score
