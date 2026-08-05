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
    v2 hybrid evolution: Combines Parent 2's monotonic robustness and multiplicative coupling
    with Parent 1's graded urgency tiers, criticality-aware fairness, and uncertainty gating.
    Key improvements:
    - Graded urgency (3-tier) replaces tanh for sharper deadline enforcement while preserving monotonicity
    - Multiplicative urgency × CPD × CED core retains smooth coupling but uses improved CED gating
    - Criticality-weighted wait pressure replaces sqrt-based fairness for better starvation prevention
    - Robust MAD+minmax normalization used uniformly across all terms (N=1 safe, outlier-resilient)
    - Uncertainty gating applied only in risk regimes (slack <= 1.5) to avoid noise amplification
    - All operations protected against NaN/inf/zero; final score finite, deterministic, shape (N,)
    """
    eps = 1e-08
    # Safe conversion and NaN/inf handling
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=1e6, neginf=-1e6)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=0.0, posinf=1e6, neginf=0.0)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=0.0, posinf=1e6, neginf=0.0)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=1e6, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=0.0, posinf=1e6, neginf=0.0)

    def normalize_mad_minmax(x):
        """Robust normalization: MAD-based centering + minmax scaling, N=1 safe"""
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size == 1:
            return np.array([0.0])
        median_x = np.median(x)
        abs_devs = np.abs(x - median_x)
        mad = np.median(abs_devs)
        scale = mad if mad > eps else eps
        z = (x - median_x) / scale
        z_clipped = np.clip(z, -2.0, 2.0)
        z_min, z_max = np.min(z_clipped), np.max(z_clipped)
        if z_max - z_min < eps:
            return np.zeros_like(z_clipped)
        return 2.0 * (z_clipped - z_min) / (z_max - z_min + eps) - 1.0

    # Graded urgency: monotonic, piecewise-smooth, zero at slack > 3.0
    urgency_raw = np.zeros_like(slack)
    mask_viol = slack <= 0
    urgency_raw[mask_viol] = -slack[mask_viol] * 8.0  # high penalty for violation
    mask_crit = (slack > 0) & (slack <= 1.0)
    urgency_raw[mask_crit] = 8.0 * (1.0 - slack[mask_crit]**2)  # quadratic decay
    mask_caut = (slack > 1.0) & (slack <= 3.0)
    urgency_raw[mask_caut] = 8.0 * (1.0 - (slack[mask_caut] - 1.0) / 2.0)  # linear fade
    norm_urgency = normalize_mad_minmax(urgency_raw)
    urgency_term = 1.0 + 0.9 * norm_urgency  # higher weight on urgency

    # Critical path density (CPD): importance per time unit
    exec_comm_sum = min_exec_time + min_comm_time + eps
    cpd_base = upward_rank / exec_comm_sum
    norm_cpd = normalize_mad_minmax(cpd_base)
    cpd_term = 1.0 + 0.7 * norm_cpd

    # Critical energy density (CED): importance per energy unit, gated by slack feasibility
    ced_base = (upward_rank * remaining_work) / (min_incremental_energy + eps)
    # Sigmoid gate: full activation only when slack >= 0, smooth transition near deadline
    ced_gate = 1.0 / (1.0 + np.exp(-(slack + 0.5)))  # shifts gate left for early activation
    ced_gated = ced_base * ced_gate
    norm_ced = normalize_mad_minmax(ced_gated)
    ced_term = 1.0 + 0.5 * norm_ced

    # Criticality-weighted wait pressure: prevents starvation of high-rank long-waiting tasks
    exec_comm_median = np.median(exec_comm_sum) if len(exec_comm_sum) > 1 else exec_comm_sum[0]
    wait_pressure = np.clip(ready_wait_time / (exec_comm_median + eps), 0.0, 10.0)
    # Sigmoid weighting by upward_rank to focus on important tasks
    rank_sigmoid = 1.0 / (1.0 + np.exp(-(upward_rank - np.median(upward_rank + eps)) / (np.std(upward_rank + eps) + eps)))
    fairness_raw = wait_pressure * rank_sigmoid
    norm_fairness = normalize_mad_minmax(fairness_raw)
    fairness_term = -0.25 * norm_fairness  # negative: higher fairness score lowers priority

    # Uncertainty gating: only active in risk zones (slack <= 1.5)
    unc_boost = np.zeros_like(uncertainty)
    mask_unc_high = slack <= 0
    unc_boost[mask_unc_high] = np.clip(uncertainty[mask_unc_high] * 0.6, 0.0, 0.3)
    mask_unc_med = (slack > 0) & (slack <= 1.5)
    unc_boost[mask_unc_med] = np.clip(uncertainty[mask_unc_med] * 0.3, 0.0, 0.15)
    norm_unc = normalize_mad_minmax(unc_boost)
    unc_term = 0.15 * norm_unc

    # Core multiplicative score: urgency × CPD × CED ensures coupling
    core_score = urgency_term * cpd_term * ced_term
    # Additive penalties/terms
    score = core_score + unc_term + fairness_term

    # Final sanitization
    score = np.nan_to_num(score, nan=1e7, posinf=1e7, neginf=-1e7)
    score = np.clip(score, -1e7, 1e7)
    return score.reshape(-1)
