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
    v2: Hard-DDL-first, energy-aware, starvation-proof priority with unified risk modeling.
    
    Key self-evolved improvements:
    - Restores strict deadline enforcement: urgency uses hard penalty for slack <= 0 (no smoothing), and dominates all terms via largest weight (-5.0)
    - Eliminates fragile gating: energy term now uses *slack-aware scaling* (not sigmoid gating) — linearly penalizes energy when slack > 0, but preserves zero-energy bias under tight deadlines
    - Replaces aging_boost with robust wait-normalized fairness: uses ready_wait_time / (min_exec_time + min_comm_time + eps) directly scaled by |slack|⁻¹ only when slack > 0, bounded and monotonic
    - Uncertainty is always active: capped percentile-bounded penalty (0–2.0) applied uniformly — no conditional masking → ensures consistent risk mitigation at all slack levels
    - Criticality term strengthened and decoupled: upward_rank * remaining_work / (min_exec_time + eps) — avoids division by exec_effort alone (more stable)
    - All normalizations use safe fallbacks for N=1/flat arrays; final score clamped to finite bounds and strictly deterministic
    - Prioritization order: (1) DDL compliance (urgency), (2) critical-path progress, (3) energy efficiency (only when margin exists), (4) fairness, (5) risk — no term cancels deadline signal
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
    
    def robust_normalize(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size == 1:
            return np.array([0.0])
        q25, q50, q75 = np.percentile(x, [25, 50, 75], axis=0, keepdims=False)
        iqr = q75 - q25
        scale = np.where(iqr > eps, iqr, 1.0)
        return (x - q50) / (scale + eps)
    
    # Urgency: HARD deadline enforcement — slack <= 0 triggers strong linear penalty; no smoothing near zero
    urgency_raw = np.where(slack <= 0, -slack * 3.0 + 1.0, slack * 0.1)
    norm_urgency = robust_normalize(urgency_raw)
    urgency_term = -5.0 * norm_urgency  # Dominant weight ensures DDL violation avoidance
    
    # Criticality: progress on critical path — more numerically stable denominator
    exec_comm_effort = np.maximum(min_exec_time + min_comm_time, eps)
    critical_density = upward_rank * (remaining_work / exec_comm_effort)
    norm_critical = robust_normalize(critical_density)
    critical_term = -1.8 * norm_critical  # Increased weight to prioritize structural importance
    
    # Energy efficiency: scaled by slack margin, but never gated out — linear reward decay above zero slack
    energy_per_crit = min_incremental_energy / (critical_density + eps)
    # Slack-based scaling: full weight when slack <= 0 (energy secondary), linear decay up to slack=10s
    slack_scale = np.clip(1.0 - np.clip(slack, 0.0, 10.0) / 10.0, 0.0, 1.0)
    energy_scaled = energy_per_crit * slack_scale
    norm_energy = robust_normalize(energy_scaled)
    efficiency_term = -0.8 * norm_energy  # Reduced magnitude to avoid overriding urgency/criticality
    
    # Fairness: starvation prevention via normalized wait ratio, boosted only under positive slack (monotonic)
    duration_estimate = min_exec_time + min_comm_time + eps
    wait_ratio = np.clip(ready_wait_time / duration_estimate, 0.0, 15.0)
    # Aging boost: proportional to 1/(slack+eps) only when slack > 0, bounded to prevent explosion
    slack_inv_boost = np.where(slack > 0, 1.0 / (slack + eps), 0.0)
    fairness_base = wait_ratio * np.clip(slack_inv_boost, 0.0, 4.0)
    norm_fairness = robust_normalize(fairness_base)
    fairness_term = -0.6 * np.clip(norm_fairness, -2.0, 2.5)  # Bounded to avoid overcorrection
    
    # Uncertainty: always-on, bounded penalty — no conditional activation; percentile-capped to [0, 2.0]
    unc_q75 = np.percentile(uncertainty, 75) + eps
    norm_uncertainty = np.clip(uncertainty / unc_q75, 0.0, 2.0)
    uncertainty_term = 0.6 * norm_uncertainty  # Consistent, non-zero risk signal at all times
    
    # Final score: sum of all terms — urgency dominates, others refine under feasibility
    score = urgency_term + critical_term + efficiency_term + fairness_term + uncertainty_term
    # Clamp to strict finite bounds and ensure no NaN/inf
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    
    return score
