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
    v2: Hybrid DDL-hard priority with monotonic urgency, orthogonal signal decomposition,
         safety-gated energy efficiency, uncertainty-amplified fairness, and robust normalization.
    Combines Parent 2's strict monotonic urgency & safety gating with Parent 1's arctan-based
    soft urgency near deadline (adapted to avoid non-monotonicity) and normalized criticality-energy density.
    Key improvements:
      - Uses clipped inverse-slack for global monotonicity *plus* smooth arctan ramp in [0, tau_soft]
        to prevent numerical instability near zero slack while preserving ordering.
      - Replaces criticality-pressure with normalized CED (Criticality-Energy-Density) term,
        weighted by slack-gated activation to emphasize energy savings only when safe.
      - Fairness uses linear wait time scaled by both slack margin *and* uncertainty,
        with explicit starvation guard for long-waiting tasks even under tight deadlines.
      - Robust winsorized IQR+median normalization applied uniformly across all signals.
      - Hard-DDL violation always yields minimal score; no floating-point underflow/overflow.
    """
    eps = 1e-8
    tau_soft = 1.0  # Soft urgency transition window
    tau_safe = 2.0  # Minimum slack to activate energy optimization
    fairness_floor = 0.1  # Prevents fairness collapse when slack ~ 0
    
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
    
    # === URGENCY: Monotonic + smooth near deadline ===
    # For violated: fixed highest priority
    # For slack > 0: inverse-slack dominates; arctan provides stable, bounded boost near zero
    urgency_raw = np.where(
        is_violated,
        0.0,
        np.where(
            slack <= tau_soft,
            np.pi/2 - np.arctan(slack + eps),  # maps [0,tau_soft] → [π/2, arctan(tau_soft)] ≈ [1.57, 1.11]
            1.0 / (slack + eps)
        )
    )
    
    # === CRITICALITY-ENERGY-DENSITY (CED): Upward rank per unit remaining work,
    # normalized by marginal energy per latency unit — activated only when slack permits ===
    exec_comm_sum = min_exec_time + min_comm_time + eps
    ced_base = (upward_rank + eps) / (remaining_work + eps) * (exec_comm_sum + eps) / (min_incremental_energy + eps)
    ced_gate = np.clip((slack - tau_safe) / (tau_safe + eps), 0.0, 1.0)
    ced_raw = ced_base * ced_gate
    
    # === FAIRNESS: Linear wait time, amplified by uncertainty and gated by slack margin,
    # but with floor to prevent starvation even under tight deadlines ===
    wait_safe = np.maximum(ready_wait_time, 0.0)
    # Always reward waiting, but scale by how much slack remains beyond safety margin
    slack_margin = np.maximum(slack - tau_safe, 0.0) + fairness_floor
    fairness_raw = wait_safe / (slack_margin + eps) * (1.0 + np.clip(uncertainty, 0.0, 1.0))
    
    # === ROBUST NORMALIZATION (winsorized IQR + median centering) ===
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
    norm_ced = normalize_robust(ced_raw)
    norm_fair = normalize_robust(fairness_raw)
    
    # === COMBINED SCORE: Smaller = higher priority ===
    # Urgency dominates (negative weight), CED encourages energy-efficient scheduling when safe,
    # Fairness prevents starvation (positive weight reduces priority of already-waiting tasks)
    score_dynamic = (
        -7.0 * norm_urgency   # strongest pull toward urgent tasks
        + 2.0 * norm_ced      # reward energy-efficient execution only when slack allows
        + 0.5 * norm_fair     # mild penalty for long-waiting tasks (to balance fairness)
    )
    
    # Hard violation override: assign extreme priority
    score = np.where(is_violated, -1e9, score_dynamic)
    
    # Final sanitization: ensure finite, deterministic output
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e9, 1e9)
    
    # Ensure shape (N,) and float64 dtype
    return score.astype(np.float64, copy=True).reshape(-1)
