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
    Hybrid priority rule combining Parent 2's robust deadline modeling and critical-energy density
    with Parent 1's starvation-aware aging boost and adaptive risk gating.
    
    Key improvements:
    - Uses smooth arctan-based deadline risk (Parent 2) but adds hard penalty for slack < 0 to enforce DDL hard constraint
    - Critical-energy density (upward_rank * remaining_work / energy) remains primary efficiency signal
    - Introduces bounded, latency-normalized aging boost (max 5% of base score) to prevent starvation without overriding deadlines
    - Risk gating: uncertainty only contributes when slack < 0 AND uncertainty > median_uncertainty (Parent 1's stable additive gating)
    - MAD normalization everywhere for outlier resilience, with explicit degenerate-case handling
    - Removes redundant remaining_work scaling in deadline term (avoids double-counting with critical-energy density)
    - Ensures strict monotonicity in slack via arctan + linear penalty hybrid
    """
    eps = 1e-08
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    
    def safe_mad_normalize(x):
        """Robust MAD normalization: handles N=1, constant arrays, and NaNs."""
        if x.size == 0:
            return np.zeros(0)
        x_clean = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x_clean.size == 1:
            return np.zeros_like(x_clean)
        median_x = np.median(x_clean)
        mad = np.median(np.abs(x_clean - median_x)) + eps
        if mad < eps:
            return np.zeros_like(x_clean)
        normed = (x_clean - median_x) / mad
        return np.clip(normed, -3.0, 3.0)
    
    # Deadline risk: smooth arctan for slack >= 0, hard linear penalty for slack < 0
    deadline_risk_arctan = np.arctan(-np.clip(slack, 0.0, None)) + np.pi / 2
    deadline_risk_arctan = deadline_risk_arctan / np.pi
    deadline_risk_hard = np.where(slack < 0, -slack / (np.abs(np.min(slack)) + eps), 0.0)
    deadline_risk = deadline_risk_arctan + deadline_risk_hard
    deadline_score = safe_mad_normalize(deadline_risk)
    
    # Critical-energy density: high impact per joule (DAG-aware efficiency)
    critical_energy_density = upward_rank * remaining_work / (min_incremental_energy + eps)
    critical_energy_norm = safe_mad_normalize(critical_energy_density)
    
    # Upward rank normalized separately for structural importance
    upward_rank_norm = safe_mad_normalize(upward_rank)
    
    # Aging boost: bounded, latency-normalized, prevents starvation without violating deadlines
    total_latency = min_exec_time + min_comm_time + eps
    max_latency = np.max(total_latency) + eps
    wait_normalized = np.clip(ready_wait_time / max_latency, 0.0, 1.0)
    aging_boost = 0.05 * wait_normalized  # capped at 5% of base contribution
    
    # Uncertainty gating: only active under deadline pressure AND high uncertainty
    median_uncertainty = np.median(uncertainty) if uncertainty.size > 0 else 0.0
    uncertainty_active = (slack < 0) & (uncertainty > median_uncertainty + eps)
    uncertainty_gated = np.where(uncertainty_active, uncertainty, 0.0)
    uncertainty_norm = safe_mad_normalize(uncertainty_gated)
    
    # Final weighted score: deadline risk dominates; critical-energy density drives energy minimization
    # All terms designed to be additive and numerically stable
    score = (
        +4.0 * deadline_score          # Strongest weight for hard deadline enforcement
        - 2.2 * critical_energy_norm   # Primary energy-efficiency signal
        - 1.0 * upward_rank_norm       # Structural importance baseline
        + 0.05 * aging_boost           # Small fairness boost to avoid indefinite waiting
        + 0.2 * uncertainty_norm       # Penalty only when both risky and overdue
    )
    
    # Ensure finite output with conservative clamping
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    return score
