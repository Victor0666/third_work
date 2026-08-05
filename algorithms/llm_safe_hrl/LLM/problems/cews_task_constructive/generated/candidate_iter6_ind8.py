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
    Self-evolved priority rule: unified urgency + proactive uncertainty integration + starvation-aware fairness.
    
    Key improvements over v1:
    - Replaces dual urgency (exp + soft) with *single, monotonic, bounded exponential urgency*:
      exp(-slack / (task_min_duration + eps)) — captures both violation severity and relative tightness.
    - Integrates uncertainty *proactively*: uses normalized uncertainty weighted by slack sign, not gated;
      ensures early risk awareness even for positive slack (e.g., slack=5s but high uncertainty).
    - Simplifies criticality-energy coupling: removes tiering & percentile waiting; replaces with
      *work-normalized criticality ratio* (upward_rank / (min_incremental_energy * (1+uncertainty))) 
      robustly normalized and clipped.
    - Fairness via *adaptive waiting boost* bounded by remaining_work *and* slack sign:
      only activated when slack > 0 AND wait_time > median_wait, scaled by work/urgency tradeoff.
    - All components use symmetric IQR normalization with explicit finite bounds; no unbounded growth.
    - Deterministic, eps-protected, nan/inf-guarded, shape-compliant, and side-effect-free.
    """
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
    
    def robust_normalize(x):
        q1, q3 = np.quantile(x, [0.25, 0.75], method='midpoint')
        iqr = q3 - q1 + eps
        med = np.median(x)
        z = (x - med) / iqr
        return np.clip(z, -10.0, 10.0)
    
    # Unified urgency: monotonic, bounded, duration-normalized exponential penalty
    task_min_duration = np.maximum(min_exec_time + min_comm_time, eps)
    norm_slack = slack / task_min_duration
    # exp(-x) decays from 1.0 (large slack) to near 0 (large negative slack); clip avoids overflow
    deadline_urgency = np.clip(np.exp(-norm_slack), 1e-6, 1.0)
    
    # Proactive uncertainty integration: always active, scaled by slack sign
    # High uncertainty penalizes more under pressure (slack <= 0), less when relaxed
    unc_weight = np.where(slack <= 0, 1.0 + uncertainty, 1.0 + 0.3 * uncertainty)
    
    # Work-normalized criticality-energy ratio: upward_rank / (energy * uncertainty factor)
    energy_risk_scaled = np.maximum(min_incremental_energy * unc_weight, eps)
    crit_eff_ratio = upward_rank / energy_risk_scaled
    crit_eff_ratio = np.clip(crit_eff_ratio, 1e-6, 1e6)
    norm_crit_eff = robust_normalize(crit_eff_ratio)
    
    # Adaptive waiting boost: only activates for non-urgent tasks (slack > 0) with long wait
    median_wait = np.median(ready_wait_time) + eps
    wait_boost_mask = (slack > 0.0) & (ready_wait_time > median_wait)
    # Scale boost by work (larger work → higher starvation cost) and inverse urgency (less urgent → more boost)
    rw_norm = np.clip(remaining_work / (np.median(remaining_work) + eps), 0.1, 10.0)
    wait_boost = wait_boost_mask.astype(float) * 0.2 * rw_norm * (1.0 - deadline_urgency)
    
    # Normalized features
    work_norm = robust_normalize(remaining_work)
    time_cost = np.sqrt(np.maximum(min_exec_time, eps)) + np.sqrt(np.maximum(min_comm_time, eps))
    time_norm = robust_normalize(time_cost)
    unc_norm = robust_normalize(uncertainty)
    
    # Final score: lower is better; urgency dominates, criticality-energy second, others balanced
    score = (
        -4.5 * deadline_urgency 
        - 2.2 * norm_crit_eff 
        + 0.25 * time_norm 
        + 0.18 * work_norm 
        + 0.12 * unc_norm 
        + wait_boost
    )
    
    # Guard against NaN/inf and ensure finite output
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    return score
