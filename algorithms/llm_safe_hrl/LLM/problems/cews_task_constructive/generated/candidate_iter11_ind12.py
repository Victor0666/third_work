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
    Priority rule v4: Lexicographic DDL-hardness + latency-aware energy density + robust fairness,
    with unified uncertainty-aware slack calibration, monotonic tie-breaking, and degenerate-safe
    normalization. Combines Parent 2's bounded urgency & LAED with Parent 1's effective_slack
    risk scaling and hard urgency fallback for strict deadline adherence.
    
    Key innovations:
      - Unified risk-calibrated slack: effective_slack = slack / (1 + uncertainty/5 + eps), used
        *both* for urgency gating *and* LAED denominator scaling → consistent DDL-risk modeling.
      - Urgency uses *hard-gated bounded sigmoid*: dominates score when effective_slack < 0,
        smoothly transitions via sigmoid when >0 but <0.1, zero otherwise → eliminates marginal slack noise.
      - LAED redefined as (upward_rank * remaining_work) / (effective_slack + min_exec_time + 
        min_comm_time + min_incremental_energy + eps), making energy efficiency explicitly reward
        tasks that reduce critical-path slack *and* consume less energy per latency unit.
      - Fairness becomes *lateness-avoiding wait pressure*: capped at |effective_slack| + eps,
        then normalized and gated by (1 - urgency_score) to prevent interference with urgent tasks.
      - All normalization uses robust_minmax_normalize with constant/N=1 fallbacks and strict clipping.
      - Tie-breaking uses *inverse latency and inverse criticality* (1/(exec+eps), 1/(rank+eps)),
        normalized and weighted lightly to break ties without overriding urgency.
      - Final score is strictly finite, lexicographically ordered (urgency >> LAED >> fairness >> tiebreak),
        with per-term clamping to preserve interpretability and numerical stability.
    """
    eps = 1e-08
    # Safe casting and NaN/inf handling
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=0.0, posinf=1e6, neginf=eps)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=0.0, posinf=1e6, neginf=eps)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=0.0, posinf=1e6, neginf=eps)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=1e6, neginf=-1e6)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=0.0, posinf=1e6, neginf=eps)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=0.0, posinf=1e6, neginf=eps)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=1e6, neginf=eps)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=0.0, posinf=1e6, neginf=eps)
    
    # Risk-calibrated effective slack: same definition used everywhere for consistency
    effective_slack = slack / (1.0 + np.clip(uncertainty / 5.0, 0.0, 100.0) + eps)
    
    def robust_minmax_normalize(x):
        """Min-max normalize to [0,1] with degenerate-case safety."""
        if x.size == 1:
            return np.full_like(x, 0.5, dtype=float)
        x_min, x_max = np.min(x), np.max(x)
        if x_max - x_min < eps:
            return np.full_like(x, 0.5, dtype=float)
        return np.clip((x - x_min) / (x_max - x_min + eps), 0.0, 1.0)
    
    # === URGENCY (lexicographically dominant) ===
    # Hard gate: top priority if effective_slack < 0
    is_urgent = effective_slack < 0.0
    # Smooth transition for small positive slack: sigmoid decay from 1→0 over [0, 0.1]
    urgency_raw = np.where(
        is_urgent,
        1.0,
        np.where(effective_slack <= 0.1,
                 1.0 / (1.0 + np.exp((effective_slack - 0.05) / 0.02)),  # centered sigmoid
                 0.0)
    )
    urgency_score = robust_minmax_normalize(urgency_raw) * 1000000.0
    
    # === LATENCY-AWARE ENERGY DENSITY (LAED) ===
    # Numerator: criticality-weighted work
    laed_numerator = upward_rank * remaining_work
    # Denominator: latency + energy + *risk-adjusted slack margin* (penalizes wasting slack)
    laed_denominator = (
        np.clip(effective_slack, 0.0, None) +  # only penalize if slack is positive
        min_exec_time + 
        min_comm_time + 
        min_incremental_energy + 
        eps
    )
    laed_raw = laed_numerator / laed_denominator
    # Gate LAED: only active when effective_slack > 0.1 (avoids noise near deadline)
    laed_gated = np.where(effective_slack > 0.1, laed_raw, 0.0)
    laed_norm = robust_minmax_normalize(laed_gated) * 100000.0
    
    # === FAIRNESS (lateness-avoiding wait pressure) ===
    # Cap wait time at deadline proximity to avoid aging bias under imminent violation
    capped_wait = np.minimum(ready_wait_time, np.abs(effective_slack) + eps)
    fairness_raw = np.sqrt(np.maximum(capped_wait, 0.0))
    fairness_max = np.max(fairness_raw) + eps
    fairness_boost = np.clip(fairness_raw / fairness_max, 0.0, 1.0)
    # Gate fairness: only apply when not urgent (prevents interference)
    fairness_gated = fairness_boost * (1.0 - urgency_score / 1000000.0)
    fairness_norm = robust_minmax_normalize(fairness_gated) * 10000.0
    
    # === TIE-BREAKING (monotonic, low-weight) ===
    # Prefer tasks with lower execution time and lower upward rank (more critical paths)
    inv_exec = 1.0 / (min_exec_time + eps)
    inv_rank = 1.0 / (upward_rank + eps)
    exec_tie = (1.0 - robust_minmax_normalize(inv_exec)) * 100.0
    rank_tie = (1.0 - robust_minmax_normalize(inv_rank)) * 10.0
    
    # === COMBINE WITH LEXICOGRAPHIC WEIGHTING AND CLAMPING ===
    score = (
        np.where(is_urgent, -1000000.0, urgency_score) +  # hard negative for urgent
        laed_norm +
        fairness_norm +
        exec_tie +
        rank_tie
    )
    
    # Final safeguard: ensure finite, deterministic output
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    
    return score.astype(float)
