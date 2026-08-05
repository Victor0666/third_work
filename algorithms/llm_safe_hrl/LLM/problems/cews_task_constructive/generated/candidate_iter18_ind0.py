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
    v2 priority rule: Restored deadline dominance with adaptive uncertainty-aware criticality & balanced starvation relief.
    
    Key improvements over v1:
    - Reinstates stronger hard-DDL bias: urgency_bias now scales linearly with -slack (not binary), preserving gradient pressure
    - Loosens critical-path gating: uses OR (tight_slack OR high_uncertainty) instead of AND to prevent under-penalizing late tasks
    - Restores fairness: work-threshold lowered to 30%-quantile (0.3) and wait-gating uses robust z-score with median-based centering
    - Uncertainty-normalized slack now applies *only* to sigmoid term — preserves crisp urgency for negative slack
    - Introduces energy-urgency coupling: energy penalty scaled by (1 - norm_urgency) to avoid over-penalizing already urgent tasks
    - Robust trimmed-minmax normalization remains (10% trim) but applied *after* all domain-aware transformations
    - All divisions guarded; NaN/inf replaced deterministically; shape-invariant; no unbounded ops
    '''
    eps = 1e-08
    min_exec_time = np.asarray(min_exec_time, dtype=float).copy()
    min_comm_time = np.asarray(min_comm_time, dtype=float).copy()
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float).copy()
    slack = np.asarray(slack, dtype=float).copy()
    upward_rank = np.asarray(upward_rank, dtype=float).copy()
    remaining_work = np.asarray(remaining_work, dtype=float).copy()
    ready_wait_time = np.asarray(ready_wait_time, dtype=float).copy()
    uncertainty = np.asarray(uncertainty, dtype=float).copy()
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=float)
    
    def robust_trimmed_minmax(x):
        if x.size == 0:
            return np.zeros_like(x)
        x = x.astype(float)
        x_sorted = np.sort(x)
        trim_n = max(1, int(0.1 * len(x_sorted)))
        x_trimmed = x_sorted[trim_n:-trim_n] if len(x_sorted) > 2 * trim_n else x_sorted
        x_min, x_max = (np.min(x_trimmed), np.max(x_trimmed))
        rng = x_max - x_min + eps
        return np.clip((x - x_min) / rng, 0.0, 1.0)
    
    task_duration = min_exec_time + min_comm_time + eps
    rel_slack = np.divide(slack, task_duration, out=np.full_like(slack, 0.0), where=task_duration != 0)
    rel_slack = np.where(np.isfinite(rel_slack), rel_slack, 0.0)
    
    # dur_uncertainty for gating and normalization
    dur_uncertainty = np.divide(uncertainty, task_duration + eps, out=np.zeros_like(uncertainty), where=task_duration + eps != 0)
    dur_uncertainty = np.where(np.isfinite(dur_uncertainty), dur_uncertainty, 0.0)
    
    # Hard-DDL bias: linear penalty for negative slack (preserves gradient, avoids step loss)
    urgency_bias = np.clip(-slack, 0.0, np.inf) / (np.median(task_duration) + eps)
    
    # Sigmoid urgency uses uncertainty-normalized slack *only* for smooth near-deadline response
    aug_rel_slack = rel_slack / (1.0 + dur_uncertainty + eps)
    finite_aug_slack = aug_rel_slack[np.isfinite(aug_rel_slack)]
    slack_center = np.mean(finite_aug_slack) if len(finite_aug_slack) > 0 else 0.0
    urgency_sigmoid = 1.0 / (1.0 + np.exp(6.0 * (aug_rel_slack - slack_center)))
    
    # Urgency combines hard bias (for lateness) + smooth sigmoid (for near-deadline); both contribute negatively to score
    urgency = urgency_sigmoid + 0.5 * urgency_bias
    
    # Lateness penalty: clipped linear penalty for negative slack
    lateness_penalty = np.where(slack < 0.0, np.clip(-slack / (np.median(task_duration) + eps), 0.0, 4.0), 0.0)
    
    # Critical-path latency penalty: gated by (tight_slack OR high_uncertainty) — restores deadline pressure
    tight_slack_mask = (rel_slack <= 0.25).astype(float)
    high_uncert_mask = (dur_uncertainty >= np.quantile(dur_uncertainty, 0.7, method='midpoint')).astype(float)
    cp_latency_gate = np.maximum(tight_slack_mask, high_uncert_mask)
    cp_latency_penalty = upward_rank * np.maximum(0.0, -slack) / (task_duration + eps) * cp_latency_gate
    
    # Energy per work, with criticality weighting
    energy_per_work = np.divide(min_incremental_energy, remaining_work + eps, out=np.full_like(min_incremental_energy, eps), where=remaining_work + eps != 0)
    crit_weight_factor = 1.0 + np.clip(upward_rank / (np.median(upward_rank + eps) + eps), 0.0, 2.0)
    base_crit_energy = energy_per_work * crit_weight_factor
    # Energy penalty attenuated for highly urgent tasks to avoid conflicting objectives
    energy_urgency_coupling = (1.0 - robust_trimmed_minmax(urgency))
    crit_weighted_energy = base_crit_energy * cp_latency_gate * energy_urgency_coupling
    
    # Starvation boost: dual-gated on wait-per-work z-score > 1.5 AND work in top 30% (restored fairness)
    wait_per_work = np.divide(ready_wait_time, remaining_work + eps, out=np.full_like(ready_wait_time, 0.0), where=remaining_work + eps != 0)
    wait_per_work = np.where(np.isfinite(wait_per_work), wait_per_work, 0.0)
    work_threshold = np.quantile(remaining_work, 0.3, method='midpoint') + eps  # restored 30% quantile
    wait_median = np.median(wait_per_work)
    wait_std = np.std(wait_per_work) + eps
    wait_z_score = np.divide(wait_per_work - wait_median, wait_std, out=np.zeros_like(wait_per_work), where=wait_std != 0)
    starvation_gate = (remaining_work >= work_threshold).astype(float) * (wait_z_score > 1.5).astype(float)
    norm_wait_time = robust_trimmed_minmax(ready_wait_time)
    wait_boost = norm_wait_time * starvation_gate * 0.25
    
    # Normalize all components robustly
    norm_urgency = robust_trimmed_minmax(urgency)
    norm_lateness = robust_trimmed_minmax(lateness_penalty)
    norm_cp_latency = robust_trimmed_minmax(cp_latency_penalty)
    norm_energy = robust_trimmed_minmax(crit_weighted_energy)
    norm_upward = robust_trimmed_minmax(upward_rank)
    norm_uncertainty = robust_trimmed_minmax(dur_uncertainty)
    
    # Final score: smaller = higher priority
    # Strongest weight on urgency (negative contribution), then lateness, energy, uncertainty, CP latency
    score = -3.2 * norm_urgency \
            + 1.4 * norm_lateness \
            + 0.9 * norm_energy \
            + 0.35 * norm_uncertainty \
            - 0.4 * norm_upward \
            + 0.25 * wait_boost \
            + 0.85 * norm_cp_latency \
            + 0.1 * urgency_bias  # retain raw bias as additive term for stability
    
    # Final sanitization
    score = np.nan_to_num(score, nan=1000000000000.0, posinf=1000000000000.0, neginf=-1000000000000.0)
    score = np.clip(score, -1000000000000.0, 1000000000000.0)
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
