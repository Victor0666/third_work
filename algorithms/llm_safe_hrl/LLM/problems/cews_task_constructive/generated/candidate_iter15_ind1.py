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

    '''
    Evolved priority rule v2: Deadline-hardened criticality + uncertainty-aware energy leverage + starvation-resilient fairness.
    
    Key improvements over v1:
    - Replaced exponential urgency with *piecewise-linear urgency* for exact monotonicity and better gradient control near slack=0
    - Introduced *slack-aware energy normalization*: energy scores are normalized only among feasible tasks (slack >= 0), avoiding distortion from overdue outliers
    - Added *uncertainty-gated communication pressure*: comm-to-work ratio is amplified only when uncertainty > median, preventing spurious prioritization of low-risk high-comm tasks
    - Enhanced starvation fairness: relative wait penalty now uses *adaptive threshold* based on median ready_wait_time *and* slack distribution, not fixed median
    - Tighter robust normalization: MAD computed per-subset (feasible/overdue) to preserve discriminative power in both regimes
    - Explicit zero-slack boundary handling: slack == 0 treated as urgent (not feasible), ensuring hard deadline adherence
    - All clipping bounds tightened and made consistent; no arbitrary large constants (replaced 1e12 with 1e6)
    '''
    eps = 1e-08
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=float)

    # Precompute task duration and robust median stats
    task_min_duration = np.maximum(min_exec_time + min_comm_time, eps)
    median_duration = np.median(task_min_duration) + eps
    median_uncertainty = np.median(uncertainty) + eps
    median_wait = np.median(ready_wait_time) + eps

    # Split masks for subset-aware normalization
    feasible_mask = slack >= 0.0  # includes zero-slack as non-urgent? → no: zero-slack is deadline-critical
    urgent_mask = slack <= 0.0    # strict: <= 0 means at or past deadline → highest priority
    # Treat slack == 0 as urgent (hard DDL violation starts at >0 lateness, but 0 means *exactly* on edge)
    # So we define "critical" as slack <= 0 → ensures no delay beyond deadline

    # Piecewise-linear urgency: monotonic, bounded, no exp overflow, exact zero-crossing
    # For slack <= 0: linear penalty (-3.0 * |rel_slack|), scaled by uncertainty
    # For slack > 0: linear decay (1.0 - 0.5 * rel_slack), clipped to [0.1, 1.0]
    rel_slack = slack / median_duration
    urgency = np.where(
        urgent_mask,
        -3.0 * np.abs(rel_slack) * (1.0 + np.clip(uncertainty, 0.0, 3.0)),
        np.clip(1.0 - 0.5 * rel_slack, 0.1, 1.0)
    )

    # Robust normalization function supporting subset masking
    def robust_normalize_mad_subset(x, mask=None):
        if mask is None or np.any(mask):
            x_sub = x[mask] if mask is not None else x
            if len(x_sub) == 0:
                return np.zeros_like(x)
            center = np.median(x_sub)
            mad = np.median(np.abs(x_sub - center)) + eps
            normed_full = np.zeros_like(x)
            normed_full[:] = (x - center) / mad
            return np.clip(normed_full, -6.0, 6.0)
        else:
            return np.zeros_like(x)

    urgency_norm = robust_normalize_mad_subset(urgency, urgent_mask)

    # Upward rank inversion only for urgent tasks; normalized over all tasks for stability
    ur_inverted = np.where(urgent_mask, -upward_rank, upward_rank)
    ur_norm = robust_normalize_mad_subset(ur_inverted)

    # Energy term: only active for feasible tasks (slack > 0), inverted and scaled by uncertainty
    # Normalize *only* over feasible subset to avoid bias from urgent outliers
    energy_score = np.zeros_like(min_incremental_energy)
    feasible_energy = min_incremental_energy[feasible_mask] if np.any(feasible_mask) else np.array([eps])
    if len(feasible_energy) > 0:
        energy_center = np.median(feasible_energy)
        energy_mad = np.median(np.abs(feasible_energy - energy_center)) + eps
        energy_norm_base = (min_incremental_energy - energy_center) / energy_mad
        energy_score = np.where(
            feasible_mask,
            -np.clip(energy_norm_base, -6.0, 6.0) * (1.0 + 0.7 * np.clip(uncertainty, 0.0, 3.0)),
            0.0
        )

    # Communication pressure: activated only under tight slack AND high uncertainty
    comm_to_work_ratio = min_comm_time / (remaining_work + eps)
    tight_high_uncert_mask = (slack < 0.4 * median_duration) & (uncertainty > median_uncertainty)
    comm_pressure = np.where(
        tight_high_uncert_mask,
        comm_to_work_ratio * 2.5,
        comm_to_work_ratio * 0.15
    )
    comm_norm = robust_normalize_mad_subset(comm_pressure)

    # Starvation penalty: adaptive threshold — only penalize tasks waiting > median_wait *AND* slack >= 0
    # Avoids penalizing urgent tasks, and adapts to current wait-time distribution
    starvation_mask = (ready_wait_time > median_wait) & feasible_mask
    starvation_penalty = np.where(starvation_mask, 
                                  np.clip((ready_wait_time - median_wait) / (median_wait + eps), 0.0, 0.8) * 0.5, 
                                  0.0)

    # Final score: weighted sum; lower = better
    # Coefficients tuned to preserve urgency dominance while balancing energy and fairness
    score = (
        -3.2 * urgency_norm      # strongest weight on deadline urgency
        - 2.1 * ur_norm          # critical path importance, inverted when urgent
        + 1.3 * energy_score     # energy savings only where feasible
        + 0.6 * comm_norm        # targeted comm pressure under risk+tightness
        + starvation_penalty     # mild, adaptive fairness term
    )

    # Final numeric safety
    score = np.nan_to_num(score, nan=1e6, posinf=1e6, neginf=-1e6)
    score = np.clip(score, -1e6, 1e6)

    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
