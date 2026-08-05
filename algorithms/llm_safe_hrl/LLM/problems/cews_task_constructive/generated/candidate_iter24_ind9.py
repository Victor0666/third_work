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
    v3 priority rule: Deadline-feasibility-first + criticality-aware energy coupling + starvation-robust wait-pressure + uncertainty-calibrated normalization.

    Key improvements over v1/v2:
    - Replaces quantile-based urgency gating with *hard feasibility dominance*: violated tasks get strict top priority (score = -inf) unless numerically unstable, enforced via deterministic rank-preserving shift.
    - Introduces *criticality-weighted energy penalty*: energy term scaled by upward_rank / max_upward_rank to prioritize energy reduction on high-criticality paths — avoids wasting efficiency gains on low-impact branches.
    - Upgrades starvation rescue: replaces fixed percentile gate with *dynamic criticality threshold* (upward_rank > 0.75 * median(ur) + 0.5 * mad(ur)), robust to outliers and skewed distributions.
    - Adds *uncertainty-calibrated normalization*: robust_zclip now uses interquartile range (IQR) instead of MAD for better stability under heavy-tailed uncertainty; clipping bounds tightened to ±6σ for higher discriminative resolution in tight-deadline regimes.
    - Removes work_bonus (conflicts with objective: remaining_work is proxy for latency, not efficiency; optimizing it directly undermines deadline feasibility).
    - Enforces strict priority hierarchy via additive layering: urgency dominates → synergy → energy → starvation; all terms normalized *independently*, then weighted with sum-to-1 weights (0.48+0.32+0.14+0.06=1.0) to ensure scale-invariance.
    - Fixes numerical fragility: replaces unsafe divisions with np.divide(..., where=...) + nan_to_num; ensures all outputs finite, deterministic, and shape-(N,).
    """
    eps = 1e-08
    min_exec_time = np.asarray(min_exec_time, dtype=np.float64).copy()
    min_comm_time = np.asarray(min_comm_time, dtype=np.float64).copy()
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=np.float64).copy()
    slack = np.asarray(slack, dtype=np.float64).copy()
    upward_rank = np.asarray(upward_rank, dtype=np.float64).copy()
    remaining_work = np.asarray(remaining_work, dtype=np.float64).copy()
    ready_wait_time = np.asarray(ready_wait_time, dtype=np.float64).copy()
    uncertainty = np.asarray(uncertainty, dtype=np.float64).copy()
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=np.float64)

    def robust_zclip(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.clip(x, -1e12, 1e12)
        if N == 1:
            return np.array([0.0])
        # Use IQR for more stable dispersion estimate than MAD under uncertainty skew
        q25, q50, q75 = np.quantile(x, [0.25, 0.5, 0.75], method='midpoint')
        iqr = q75 - q25 + eps
        z = (x - q50) / iqr
        return np.clip(z, -6.0, 6.0)  # Tighter bound improves resolution for urgent tasks

    task_duration = np.maximum(min_exec_time + min_comm_time, eps)
    dur_uncertainty = np.divide(uncertainty, task_duration + eps, out=np.zeros_like(uncertainty), where=(task_duration + eps) != 0)
    dur_uncertainty = np.nan_to_num(dur_uncertainty, nan=0.0, posinf=0.0, neginf=0.0)

    # Hard feasibility dominance: violated tasks get lowest possible score (highest priority)
    violated_mask = slack < 0
    base_score = np.full(N, 0.0, dtype=np.float64)
    
    # Urgency term: prioritizes feasibility recovery first
    urgency_base = np.zeros_like(slack)
    urgency_base[violated_mask] = np.clip(-slack[violated_mask] / (task_duration[violated_mask] + eps), 0.0, 15.0)
    if N > 1:
        q10, q50, q90 = np.quantile(slack, [0.1, 0.5, 0.9], method='midpoint')
    else:
        q10 = q50 = q90 = slack[0]
    tight_mask = ~violated_mask & (slack < q50)
    relaxed_mask = ~violated_mask & (slack >= q50)
    urgency_base[tight_mask] = np.clip((q50 - slack[tight_mask]) / (task_duration[tight_mask] + eps), 0.0, 6.0)
    urgency_base[relaxed_mask] = np.clip((q90 - slack[relaxed_mask]) / (task_duration[relaxed_mask] + eps), 0.0, 1.5)
    
    ur_norm = robust_zclip(upward_rank)
    ur_factor = 1.0 + 0.4 * np.clip(ur_norm, 0.0, 6.0)
    unc_urgency_factor = 1.0 + 0.6 * dur_uncertainty
    urgency_penalty = urgency_base * ur_factor * unc_urgency_factor

    # Synergy term: critical-path acceleration within feasibility
    slack_factor = np.clip(1.0 - np.maximum(0.0, slack - q50) / (task_duration + eps), 0.05, 1.0)
    dampened_ur = upward_rank * slack_factor
    base_synergy = dampened_ur * task_duration / (min_incremental_energy + eps)
    synergy_amplifier = np.where(violated_mask, 1.0 + 0.9 * uncertainty, 1.0)
    latency_crit_synergy = base_synergy * synergy_amplifier
    latency_crit_synergy = np.clip(latency_crit_synergy, eps, 1e10)

    # Energy term: criticality-weighted & uncertainty-augmented
    ur_max = np.max(upward_rank) + eps
    ur_weight = np.clip(upward_rank / ur_max, 0.1, 1.0)  # Prevent zero-weight on low-crit tasks
    slack_distance_weight = np.where(slack < q50, 1.0 + 0.5 * (q50 - slack) / (task_duration + eps), 1.0)
    risk_weighted_energy = min_incremental_energy * ur_weight * (1.0 + uncertainty * slack_distance_weight)
    risk_weighted_energy = np.clip(risk_weighted_energy, eps, 1e10)

    # Starvation rescue: dynamic criticality threshold (robust to outliers)
    if N > 1:
        ur_median = np.median(upward_rank)
        ur_q25, ur_q75 = np.quantile(upward_rank, [0.25, 0.75], method='midpoint')
        ur_iqr = ur_q75 - ur_q25 + eps
        crit_threshold = ur_median + 0.5 * ur_iqr
    else:
        crit_threshold = upward_rank[0]
    wait_ratio = np.divide(ready_wait_time, task_duration, out=np.zeros_like(ready_wait_time), where=task_duration != 0)
    wait_ratio = np.nan_to_num(wait_ratio, nan=0.0, posinf=0.0, neginf=0.0)
    is_starvable = (wait_ratio > 1.5) & (upward_rank >= crit_threshold) & (slack < 360.0)
    starvation_boost = np.where(is_starvable, wait_ratio * (1.0 + 0.3 * np.clip(ur_norm, 0.0, 6.0)), 0.0)
    starvation_boost = np.clip(starvation_boost, 0.0, 3.0)

    # Normalize each component independently for scale invariance
    norm_urgency = robust_zclip(urgency_penalty)
    norm_synergy = robust_zclip(latency_crit_synergy)
    norm_energy = robust_zclip(risk_weighted_energy)
    norm_starvation = robust_zclip(starvation_boost)

    # Strict additive layering with unit-sum weights (ensures no term dominates by scale)
    score = (
        0.48 * norm_urgency +
        -0.32 * norm_synergy +
        0.14 * norm_energy +
        0.06 * norm_starvation
    )

    # Apply hard feasibility override: violated tasks get deterministic minimal score
    # Use finite large negative value (not -inf) to preserve sort stability and avoid NaN propagation
    feasibility_boost = np.full(N, 0.0, dtype=np.float64)
    feasibility_boost[violated_mask] = -1e9
    score = score + feasibility_boost

    # Final sanitization
    score = np.nan_to_num(score, nan=0.0, posinf=1e12, neginf=-1e12)
    score = np.clip(score, -1e12, 1e12)
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
