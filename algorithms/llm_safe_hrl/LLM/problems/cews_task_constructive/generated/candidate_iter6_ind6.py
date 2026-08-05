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
    Self-evolved priority rule: restores urgency for near-critical tasks,
    replaces geometric efficiency mean with robust harmonic energy-work ratio,
    and inverts uncertainty damping to *amplify* attention on uncertain late tasks.
    
    Key improvements over v1:
    - Replaces restrictive critical-path gating with smooth urgency ramp: 
      urgency = sigmoid(-slack / (task_min_duration + eps)) → captures near-deadline risk without hard threshold
    - Harmonic energy efficiency: 2 / (1/energy_per_work + 1/work_efficiency_ratio) → 
      stable under low-energy/low-work outliers; avoids division-by-zero via eps-clipping
    - Uncertainty *amplification*: late tasks (slack <= 0) get upward_rank scaled by (1 + uncertainty)^gamma, 
      where gamma = clip(1.0 + 0.8 * max(0, -slack), 1.0, 4.0) → aligns with risk objective
    - Starvation guard refined: activates only when (ready_wait_time > p90) AND (remaining_work > p10) → 
      prevents trivial wait dominance while ensuring fairness for genuinely starved high-work tasks
    - Unified normalization uses median-IQR with variance fallback and tighter clipping [-8,8] for stability
    - All operations eps-protected, nan/inf guarded, deterministic, and shape-compliant
    """
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
    
    def robust_normalize(x):
        if x.size == 0:
            return np.zeros_like(x)
        med = np.median(x)
        q1, q3 = np.quantile(x, [0.25, 0.75], method='midpoint')
        iqr = q3 - q1 + eps
        # Fallback to constant if IQR near zero
        iqr = np.where(iqr < eps, 1.0, iqr)
        norm = (x - med) / iqr
        return np.clip(norm, -8.0, 8.0)
    
    task_min_duration = np.maximum(min_exec_time + min_comm_time, eps)
    
    # Smooth urgency: sigmoid of normalized lateness → captures near-critical risk
    normalized_lateness = -slack / (task_min_duration + eps)
    deadline_urgency = 1.0 / (1.0 + np.exp(-normalized_lateness))
    
    # Risk-adjusted energy with aggressive amplification for late+uncertain tasks
    risk_exponent = np.clip(1.0 + 0.8 * np.maximum(0.0, -slack), 1.0, 4.0)
    energy_risk_weighted = min_incremental_energy * np.power(1.0 + uncertainty, risk_exponent)
    energy_safe = np.maximum(energy_risk_weighted, eps)
    
    # Robust harmonic energy efficiency: avoids outlier sensitivity of geometric mean
    energy_per_work = min_incremental_energy / (remaining_work + eps)
    work_efficiency_ratio = remaining_work / (min_incremental_energy + eps)
    # Harmonic mean: 2ab/(a+b) = 2 / (1/a + 1/b)
    harmonic_eff = 2.0 / (1.0 / (energy_per_work + eps) + 1.0 / (work_efficiency_ratio + eps))
    harmonic_eff = np.clip(harmonic_eff, 1e-06, 1e6)
    norm_energy_eff = robust_normalize(harmonic_eff)
    
    # Uncertainty-amplified criticality: late + uncertain tasks get higher rank weight
    amp_factor = np.power(1.0 + uncertainty, risk_exponent)
    amplified_upward_rank = upward_rank * amp_factor
    crit_eff_ratio = amplified_upward_rank / energy_safe
    crit_eff_ratio = np.clip(crit_eff_ratio, 1e-06, 1e6)
    crit_eff_norm = robust_normalize(crit_eff_ratio)
    
    # Refined starvation guard: only triggers for high-wait AND non-trivial work
    p90_wait = np.percentile(ready_wait_time, 90, method='midpoint') if N > 1 else np.max(ready_wait_time)
    p10_work = np.percentile(remaining_work, 10, method='midpoint') if N > 1 else np.min(remaining_work)
    starvation_cond = (ready_wait_time > p90_wait + eps) & (remaining_work > p10_work + eps)
    starvation_penalty = np.where(starvation_cond, robust_normalize(ready_wait_time), 0.0)
    
    # Time cost and other normalized features
    time_cost = np.sqrt(np.maximum(min_exec_time, eps)) + np.sqrt(np.maximum(min_comm_time, eps))
    time_norm = robust_normalize(time_cost)
    unc_norm = robust_normalize(uncertainty)
    work_norm = robust_normalize(remaining_work)
    
    # Final weighted score: smaller = higher priority
    score = (
        -4.0 * deadline_urgency          # Strongest weight on deadline feasibility
        - 2.0 * crit_eff_norm           # Criticality-energy tradeoff
        + 0.5 * norm_energy_eff         # Positive weight on efficiency (lower is better, so + promotes it)
        + 0.35 * time_norm              # Mild penalty on long-duration tasks
        + 0.25 * work_norm              # Mild penalty on large remaining work (to avoid biasing toward tiny tasks)
        + 0.2 * unc_norm                # Mild penalty on high uncertainty
        + 0.25 * starvation_penalty     # Fairness-preserving penalty for truly starved high-work tasks
    )
    
    # Final numeric safeguard
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    return score
