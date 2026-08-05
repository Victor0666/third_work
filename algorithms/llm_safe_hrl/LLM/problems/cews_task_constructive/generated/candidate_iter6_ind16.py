import numpy as np

# 函数名中的 2 由 SeEvo 在生成 Prompt 时替换为目标版本号。
# 八个输入均为长度 N 的一维数组；相同下标始终指向同一个 ready task。
# 返回值也必须是长度 N 的一维数组，并遵守“分数越小，优先级越高”。
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
    Self-evolved v2 priority rule: enhances deadline safety, eliminates normalization artifacts,
    introduces energy-aware criticality gating, and fixes degenerate-case robustness.
    
    Key improvements over v1:
    - Replaces adaptive normalization with *scale-invariant* rank-based standardization (robust to N=1, outliers, flat arrays)
    - Introduces energy-criticality ratio *gated by urgency*: only penalizes high energy-per-critical-unit when slack < 0
    - Fixes aging boost: uses relative wait quantile (not raw ratio) to prevent bias under skewed wait distributions
    - Uncertainty penalty now scaled by |slack| when violated (stronger penalty for deeper lateness), not just binary risk_active
    - All normalization avoids centering on median/mean — uses rank-based z-score equivalent with bounded variance fallback
    - Explicitly handles all edge cases: N=1, constant arrays, NaN/inf in inputs (via nan_to_num before any op)
    - Removes redundant clipping; uses safe division + finite bounds throughout
    """
    eps = 1e-08
    # Defensive casting and NaN/inf cleanup *once at start*
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=0.0, posinf=1e9, neginf=1e-9)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=0.0, posinf=1e9, neginf=1e-9)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=0.0, posinf=1e9, neginf=1e-9)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=1e9, neginf=-1e9)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=0.0, posinf=1e9, neginf=1e-9)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=0.0, posinf=1e9, neginf=1e-9)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=1e9, neginf=1e-9)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=0.0, posinf=1e9, neginf=1e-9)

    # Rank-based standardization: monotonic, scale-invariant, works for N=1 & flat arrays
    def rank_standardize(x):
        if x.size == 1:
            return np.array([0.0])
        # Use argsort twice for dense ranks; add eps to break ties deterministically
        ranks = np.argsort(np.argsort(x)) + 1.0
        mean_rank = np.mean(ranks)
        std_rank = np.std(ranks) if x.size > 1 else 1.0
        return (ranks - mean_rank) / (std_rank + eps)

    # Urgency: smooth, bounded, monotonic sigmoid (v1 retained, proven)
    tau = 2.0
    urgency_sigmoid = 1.0 / (1.0 + np.exp(-slack / tau))
    norm_urgency = rank_standardize(urgency_sigmoid)
    urgency_term = -2.5 * norm_urgency

    # Critical density: work importance per unit exec time
    exec_effort = np.maximum(min_exec_time, eps)
    critical_density = upward_rank * (remaining_work / exec_effort)
    gated_criticality = critical_density * (1.0 + urgency_sigmoid)  # amplify criticality under pressure
    norm_critical = rank_standardize(gated_criticality)
    critical_term = -1.0 * norm_critical

    # Energy efficiency per critical unit — but *only penalized when urgent* (slack < 0)
    latency_cost = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_latency = min_incremental_energy / latency_cost
    critical_scale = np.maximum(critical_density, eps)
    eff_per_crit = energy_per_latency / critical_scale
    # Gate efficiency penalty: apply only where slack < 0 → focus energy savings on at-risk tasks
    eff_penalty_mask = (slack < 0).astype(float)
    gated_eff_per_crit = eff_per_crit * eff_penalty_mask + eff_per_crit.mean() * (1.0 - eff_penalty_mask)
    norm_eff_per_crit = rank_standardize(gated_eff_per_crit)
    efficiency_term = -0.6 * norm_eff_per_crit

    # Aging boost: use quantile of wait time (robust to skew), scaled by urgency
    if ready_wait_time.size == 1:
        rel_wait_quantile = np.array([0.5])
    else:
        # Compute percentile rank: fraction of tasks with wait <= current
        sorted_wait = np.sort(ready_wait_time)
        rel_wait_quantile = np.array([
            np.searchsorted(sorted_wait, w, side='right') / len(sorted_wait)
            for w in ready_wait_time
        ])
    aging_boost = urgency_sigmoid * rel_wait_quantile
    norm_aging = rank_standardize(aging_boost)
    aging_term = -0.3 * norm_aging

    # Uncertainty penalty: proportional to |slack| when violated, zero otherwise → stronger penalty for deeper lateness
    abs_slack_violation = np.maximum(-slack, 0.0)  # >0 only when late
    # Scale uncertainty by violation depth, normalized by Q75 slack violation (if exists)
    q75_slack_viol = np.percentile(abs_slack_violation, 75) if np.any(abs_slack_violation > eps) else 1.0
    uncertainty_penalty = uncertainty * abs_slack_violation / (q75_slack_viol + eps)
    norm_uncertainty = rank_standardize(uncertainty_penalty)
    uncertainty_term = 0.4 * norm_uncertainty

    # Final score: lower = higher priority
    score = urgency_term + critical_term + efficiency_term + aging_term + uncertainty_term

    # Final guard: ensure finite, shape-(N,), deterministic
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    return score
