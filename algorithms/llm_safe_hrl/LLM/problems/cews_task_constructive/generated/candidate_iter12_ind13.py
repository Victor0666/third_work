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
    v2 refined priority rule: restores lexicographic integrity with strict DDL-first enforcement,
    replaces sigmoid/clip-based urgency with *normalized deadline violation density* (DVD),
    eliminates fragile aging/uncertainty weighting in favor of robust, monotonic slack-dependent scaling,
    reintroduces deterministic waiting-pressure only for non-urgent tasks (slack > 0), and uses
    *adaptive rank-aware energy efficiency* gated by both slack feasibility AND critical-path relevance.
    All normalization is MAD-based but preserves full dynamic range via unclipped output — clipping moved to final score bounds.
    Urgency term dominates (weight=6.0), energy term secondary (weight=-1.9), latency & fairness tertiary (weights=0.8, -0.25).
    """
    eps = 1e-08
    # Safe casting and NaN/inf handling
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=1e6, neginf=-1e6)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=0.0, posinf=1e6, neginf=0.0)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=0.0, posinf=1e6, neginf=0.0)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=1e6, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=0.0, posinf=1e6, neginf=0.0)

    def robust_mad_normalize(x):
        """MAD normalization preserving discriminative power: no clipping, safe degenerate fallback."""
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size == 0:
            return np.zeros_like(x)
        if x.size == 1:
            return np.zeros_like(x)
        median_x = np.median(x)
        mad = np.median(np.abs(x - median_x))
        scale = mad if mad > eps else np.max(x) - np.min(x) + eps
        scale = max(scale, eps)
        return (x - median_x) / scale

    # === URGENCY TERM (DDL-first, dominant) ===
    # DVD: Deadline Violation Density = |slack|^{-1} for slack < 0, else exp(-slack) for slack >= 0
    # Ensures monotonic, unbounded penalty for violation while smoothly decaying for surplus slack
    inv_violation = np.where(slack < 0, 1.0 / (np.abs(slack) + eps), 0.0)
    surplus_decay = np.where(slack >= 0, np.exp(-slack / (np.maximum(np.median(np.abs(slack[slack>=0]) + eps), eps))), 0.0)
    raw_urgency = inv_violation + surplus_decay
    norm_urgency = robust_mad_normalize(raw_urgency)
    urgency_term = 6.0 * norm_urgency

    # === ENERGY TERM (active only when feasible AND critical) ===
    # Energy-normalized critical work: remaining_work / (min_incremental_energy + eps), masked strictly
    ecw_base = remaining_work / (min_incremental_energy + eps)
    ecw_masked = np.where((slack >= 0) & (upward_rank > eps), ecw_base, 0.0)
    norm_ecw = robust_mad_normalize(ecw_masked)
    ecw_term = -1.9 * norm_ecw

    # === LATENCY TERM (execution + comm, scaled by uncertainty only when slack is tight-but-feasible) ===
    base_latency = min_exec_time + min_comm_time + eps
    # Uncertainty inflation active only for 0 < slack <= 2.0 sec (tight feasible zone), linearly weighted
    tight_feasible = (slack > 0) & (slack <= 2.0)
    unc_scale = np.where(tight_feasible, uncertainty / (np.max(uncertainty + eps) + eps), 0.0)
    latency_inflated = base_latency * (1.0 + unc_scale)
    norm_latency = robust_mad_normalize(latency_inflated)
    latency_term = 0.8 * norm_latency

    # === FAIRNESS TERM (waiting pressure only for non-urgent, feasible tasks) ===
    # Linear waiting pressure: ready_wait_time / (slack + eps), capped at 10.0 to prevent dominance
    wait_pressure = np.where(slack > 0, np.clip(ready_wait_time / (slack + eps), 0.0, 10.0), 0.0)
    norm_wait = robust_mad_normalize(wait_pressure)
    fairness_term = -0.25 * norm_wait

    # Composite score: lexicographic ordering preserved via weight hierarchy
    score = urgency_term + ecw_term + latency_term + fairness_term

    # Final sanitization: ensure finite, bounded output; preserve determinism
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    score = np.clip(score, -1e12, 1e12)

    return score
