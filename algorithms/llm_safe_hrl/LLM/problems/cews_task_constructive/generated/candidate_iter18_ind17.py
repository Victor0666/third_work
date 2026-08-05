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

    eps = 1e-08
    def sanitize(x):
        x = np.asarray(x, dtype=float)
        return np.nan_to_num(x, nan=eps, posinf=1000000000000.0, neginf=eps)
    min_exec_time = sanitize(min_exec_time)
    min_comm_time = sanitize(min_comm_time)
    min_incremental_energy = sanitize(min_incremental_energy)
    slack = sanitize(slack)
    upward_rank = sanitize(upward_rank)
    remaining_work = sanitize(remaining_work)
    ready_wait_time = sanitize(ready_wait_time)
    uncertainty = sanitize(uncertainty)
    
    # Robust slack: subtract uncertainty margin but cap extreme values
    robust_slack = slack - 2.0 * uncertainty
    robust_slack = np.clip(robust_slack, -1000000000000.0, 1000000000000.0)
    
    # Smooth, strictly monotonic deadline penalty: quadratic ramp + linear near-deadline buffer
    # Eliminates flat regions and discontinuities; ensures strict urgency ordering
    deadline_penalty = np.maximum(0.0, -robust_slack) ** 2 + 0.3 * np.maximum(0.0, -robust_slack + 0.05)
    
    # Soft energy gating: exponential decay from 1.0 (slack > 0.1) to ~0.1 (slack ≈ -0.2)
    # Replaces hard binary gate → enables gradual energy de-prioritization near deadline
    slack_normalized = np.clip(robust_slack / (np.abs(robust_slack) + eps), -1.0, 1.0)
    energy_gate = np.exp(-np.maximum(0.0, -robust_slack) / (0.1 + eps))
    
    # Critical-energy density: marginal energy per unit urgency-adjusted latency
    # Restores urgency-weighted efficiency signal lost in v1's raw SEER ratio
    total_latency = min_exec_time + min_comm_time + eps
    critical_energy_density = (min_incremental_energy + eps) / (total_latency + eps) * energy_gate
    
    # Uncertainty-weighted critical path importance: suppresses low-confidence paths more aggressively
    cp_density = (upward_rank + eps) * (remaining_work + eps) / (1.0 + uncertainty ** 2 + eps)
    
    # Latency-aware fairness: activated only when safe AND waiting significantly longer than exec+comm
    wait_ratio = ready_wait_time / (total_latency + eps)
    fairness_boost = np.where(
        (robust_slack > 0.05) & (wait_ratio > 0.4),
        np.clip(0.12 * (wait_ratio - 0.4), 0.0, 0.12),
        0.0
    )
    
    # Risk penalty: only for truly late *and* uncertain tasks — avoids over-penalizing low-uncertainty lateness
    uncertainty_risk = np.where(
        (robust_slack < -0.1) & (uncertainty > 0.15),
        np.clip(uncertainty * np.abs(robust_slack) ** 0.8, 0.0, 0.6),
        0.0
    )
    
    # MAD-based normalization with degenerate handling for singleton/small arrays
    def safe_mad_normalize(x):
        if x.size == 0:
            return np.zeros_like(x)
        if x.size == 1:
            return np.zeros_like(x)
        median_x = np.median(x)
        mad = np.median(np.abs(x - median_x)) + eps
        if mad < eps:
            return np.zeros_like(x)
        normed = (x - median_x) / mad
        return np.clip(normed, -2.0, 2.0)
    
    deadline_norm = safe_mad_normalize(deadline_penalty)
    energy_norm = safe_mad_normalize(critical_energy_density)
    cp_norm = safe_mad_normalize(cp_density)
    fairness_norm = safe_mad_normalize(fairness_boost)
    risk_norm = safe_mad_normalize(uncertainty_risk)
    
    # Balanced weighting: reduced deadline dominance (+6.5 vs +8.0) prevents panic scheduling;
    # increased energy weight (-3.8 vs -4.0) preserves efficiency signal under moderate slack;
    # lower fairness weight (+0.08) reduces noise while preserving starvation mitigation;
    # risk penalty scaled down (+0.2) avoids premature migration of borderline late tasks
    score = (
        +6.5 * deadline_norm 
        - 3.8 * energy_norm 
        - 1.8 * cp_norm 
        + 0.08 * fairness_norm 
        + 0.2 * risk_norm
    )
    
    # Final sanitization: ensure finite, shape-(N,) output
    score = np.clip(score, -1000000000000.0, 1000000000000.0)
    score = np.nan_to_num(score, nan=1000000000000.0, posinf=1000000000000.0, neginf=-1000000000000.0)
    return score.reshape(-1)
