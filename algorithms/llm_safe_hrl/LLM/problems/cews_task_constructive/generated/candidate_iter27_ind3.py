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
    v5 priority rule: Hard-feasibility-first + monotonic slack-dominant urgency + 
                      criticality-robust starvation + uncertainty-aware synergy scaling +
                      bounded lateness amplification with slack-distance fidelity.

    Key improvements:
    - Replaces unstable -inv_slack with direct *slack distance* (q50 - slack) for monotonic, discriminative urgency near deadline;
      preserves linear dominance while avoiding singularity and inversion loss at slack≈0.
    - Fixes starvation threshold: uses median + 0.5*IQR (restored from v3) for robustness to skew; adds slack < 300s guard (tighter than v4's 600s) to prioritize rescue only under tight feasibility pressure.
    - Bounded lateness amplification: replaces unclipped linear penalty with saturating arctan-based penalty (≈linear for |slack|<100s, bounded by ±1e12), ensuring stability and constraint fidelity.
    - Synergy term now uses *uncertainty-calibrated denominator*: min_incremental_energy * (1 + 0.7*uncertainty) instead of raw energy — prevents over-prioritizing low-energy tasks with high risk.
    - All normalization uses IQR-based robust_zclip with ±6σ; all divisions guarded; all outputs finite, deterministic, shape-(N,).
    - Layered weights sum to 1.0: 0.52 urgency (enhanced fidelity), 0.23 synergy (risk-coupled), 0.15 energy (criticality-weighted), 0.1 starvation (tight-schedule focused).
    '''
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
        q25, q50, q75 = np.quantile(x, [0.25, 0.5, 0.75], method='midpoint')
        iqr = q75 - q25 + eps
        z = (x - q50) / iqr
        return np.clip(z, -6.0, 6.0)
    
    task_duration = np.maximum(min_exec_time + min_comm_time, eps)
    violated_mask = slack < 0
    
    # Feasibility boost: deterministic hard priority for violated tasks
    feasibility_boost = np.full(N, 0.0, dtype=np.float64)
    feasibility_boost[violated_mask] = -1e12
    
    # Bounded lateness amplification: arctan-based to avoid unbounded growth, monotonic, stable
    lateness_penalty = np.where(
        violated_mask,
        -1e12 + 1e9 * np.arctan(np.abs(slack) / 10.0),  # saturates ~±1.57e9, avoids linear explosion
        0.0
    )
    
    # Monotonic urgency: slack distance from median — preserves ordering near zero slack, no singularity
    if N > 1:
        q50_slack = np.quantile(slack, 0.5, method='midpoint')
        urgency_base = np.clip(q50_slack - slack, 0.0, 1e6)  # positive → smaller slack → higher urgency
    else:
        urgency_base = np.clip(-slack, 0.0, 1e6)
    
    # Uncertainty scaling only for non-violated tasks (violated already boosted)
    unc_urgency_factor = np.where(violated_mask, 1.0, 1.0 + 0.6 * uncertainty)
    urgency_penalty = urgency_base * unc_urgency_factor
    
    # Criticality normalization and synergy
    ur_max = np.max(upward_rank) + eps
    ur_normalized = np.clip(upward_rank / ur_max, 0.1, 1.0)
    # Synergy: criticality × duration / (risk-adjusted energy)
    risk_adj_energy = min_incremental_energy * (1.0 + 0.7 * uncertainty)
    base_synergy = np.divide(
        ur_normalized * task_duration,
        risk_adj_energy + eps,
        out=np.zeros_like(ur_normalized),
        where=(risk_adj_energy + eps) != 0
    )
    synergy_amplifier = np.where(violated_mask, 1.0 + 0.8 * uncertainty, 1.0)
    latency_crit_synergy = base_synergy * synergy_amplifier
    latency_crit_synergy = np.clip(latency_crit_synergy, eps, 1e10)
    
    # Energy term: criticality-weighted & uncertainty-penalized energy density
    energy_density = np.divide(
        min_incremental_energy,
        task_duration + eps,
        out=np.zeros_like(min_incremental_energy),
        where=(task_duration + eps) != 0
    )
    energy_density_adj = energy_density * ur_normalized * (1.0 + 0.5 * uncertainty)
    energy_density_adj = np.clip(energy_density_adj, eps, 1e10)
    
    # Robust starvation detection: median + 0.5*IQR (restored robustness), tighter slack guard
    if N > 1:
        ur_median = np.median(upward_rank)
        ur_q25, ur_q75 = np.quantile(upward_rank, [0.25, 0.75], method='midpoint')
        ur_iqr = ur_q75 - ur_q25 + eps
        crit_threshold = ur_median + 0.5 * ur_iqr
    else:
        crit_threshold = upward_rank[0]
    
    wait_ratio = np.divide(
        ready_wait_time,
        task_duration + eps,
        out=np.zeros_like(ready_wait_time),
        where=(task_duration + eps) != 0
    )
    wait_ratio = np.nan_to_num(wait_ratio, nan=0.0, posinf=0.0, neginf=0.0)
    is_starvable = (wait_ratio > 1.8) & (upward_rank >= crit_threshold) & (slack < 300.0)
    starvation_boost = np.where(
        is_starvable,
        wait_ratio * ur_normalized * (1.0 + 0.25 * uncertainty),
        0.0
    )
    starvation_boost = np.clip(starvation_boost, 0.0, 2.5)
    
    # Normalize components independently
    norm_urgency = robust_zclip(urgency_penalty)
    norm_synergy = robust_zclip(latency_crit_synergy)
    norm_energy = robust_zclip(energy_density_adj)
    norm_starvation = robust_zclip(-starvation_boost)  # negative → boost lowers score
    
    # Weighted composition (sum-to-1)
    score = (
        0.52 * norm_urgency +
        0.23 * norm_synergy +
        0.15 * norm_energy +
        0.10 * norm_starvation
    )
    
    # Apply feasibility and lateness signals
    score = score + feasibility_boost + lateness_penalty
    
    # Final sanitization
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    score = np.clip(score, -1e12, 1e12)
    
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
