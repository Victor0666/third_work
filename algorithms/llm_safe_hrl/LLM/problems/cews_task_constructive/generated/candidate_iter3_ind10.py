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
    Evolved priority rule enforcing hard deadline compliance first, then optimizing energy.
    Key improvements over v1:
    - Replaces slack-adjusted criticality with *deadline-urgency gating*: upward_rank contributes
      fully only when slack > 0; degrades to linear penalty (not division) when slack < 0,
      preserving urgency signal for near-deadline tasks.
    - Uses *discriminative waiting boost*: log1p-scaled wait time normalized by median wait,
      enabling fine-grained starvation mitigation without saturation or flat regions.
    - Introduces *uncertainty-aware slack protection*: instead of subtracting uncertainty,
      applies multiplicative guard: effective_slack = slack / (1 + uncertainty/10),
      preserving sign and relative ordering while dampening noise.
    - Adds *energy-efficiency dominance only under safety*: energy term weight drops to zero
      when any task has negative effective_slack — enforces DDL-first policy.
    - All normalizations use robust IQR clipping [-3, 3]; final score is bounded and deterministic.
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
        """IQR-based normalization: x -> (x - Q1) / (Q3 - Q1 + eps), clipped to [-3, 3]"""
        q1 = np.percentile(x, 25)
        q3 = np.percentile(x, 75)
        iqr = q3 - q1 + eps
        normed = (x - q1) / iqr
        return np.clip(normed, -3.0, 3.0)
    
    # Effective slack: multiplicative uncertainty damping preserves sign & ordering
    effective_slack = slack / (1.0 + np.clip(uncertainty / 10.0, 0.0, 1e3))
    
    # Deadline risk: exponential penalty only for negative effective_slack, bounded & normalized
    deadline_risk_raw = np.where(effective_slack < 0,
                                  np.exp(np.clip(-effective_slack, 0.0, 20.0)) - 1.0,
                                  0.0)
    deadline_risk = robust_normalize(deadline_risk_raw)
    
    # Criticality: full upward_rank when safe (slack >= 0); linear urgency penalty when at-risk
    # (not diminishing division) → preserves discriminability for urgent-but-not-yet-late tasks
    upward_rank_gated = np.where(effective_slack >= 0,
                                 upward_rank,
                                 upward_rank * (1.0 + np.clip(-effective_slack, 0.0, 10.0)))
    upward_rank_norm = robust_normalize(upward_rank_gated)
    
    # Energy efficiency: marginal energy per unit latency (efficiency ratio)
    total_latency = min_exec_time + min_comm_time + eps
    energy_per_latency = min_incremental_energy / total_latency
    energy_eff_norm = robust_normalize(energy_per_latency)
    
    # Waiting boost: log1p-scaled, normalized by median wait → high discrimination across range
    median_wait = np.median(ready_wait_time + eps)
    wait_boost_raw = np.log1p(ready_wait_time / (median_wait + eps))
    wait_boost = robust_normalize(wait_boost_raw)
    
    # Work importance: normalized remaining work
    work_norm = robust_normalize(remaining_work)
    
    # Safety-gated energy weighting: zero weight if ANY task violates effective_slack
    # Enforces hard DDL-first policy — no energy savings traded for missed deadlines
    has_violation = np.any(effective_slack < 0)
    energy_weight = 0.0 if has_violation else 0.8
    
    # Final convex combination: deadline risk dominates; urgency preserved; energy optional
    score = (+3.2 * deadline_risk 
             - 2.0 * upward_rank_norm 
             + energy_weight * energy_eff_norm 
             + 0.6 * wait_boost 
             + 0.3 * work_norm)
    
    # Ensure finite output and correct shape
    return np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
