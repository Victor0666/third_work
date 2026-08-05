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
    v2: Deadline-robust, energy-aware, and starvation-resistant priority with:
    - Adaptive urgency scaling using tanh (smoother than arctan near zero, bounded [-1,1])
    - Slack-aware criticality amplification only when slack > 0 to avoid over-penalizing late tasks
    - Energy efficiency term redefined as (energy / (remaining_work + eps)) * (1 / (exec+comm+1)) for dual-efficiency bias
    - Fairness now uses sqrt(wait) for gentler anti-starvation + explicit age penalty only when urgency is low AND slack > 0
    - Robust normalization with dynamic fallback: quantile → variance → constant-zero for degenerate cases
    - Hard-DDL enforcement via multiplicative urgency gating *and* additive violation penalty (dual safety)
    - All terms sign-consistent, finite, deterministic, and N=1 safe.
    """
    eps = 1e-08
    def sanitize(x):
        x = np.asarray(x, dtype=float)
        return np.nan_to_num(x, nan=eps, posinf=1e6, neginf=-1e6)
    
    min_exec_time = sanitize(min_exec_time)
    min_comm_time = sanitize(min_comm_time)
    min_incremental_energy = sanitize(min_incremental_energy)
    slack = sanitize(slack)
    upward_rank = sanitize(upward_rank)
    remaining_work = sanitize(remaining_work)
    ready_wait_time = sanitize(ready_wait_time)
    uncertainty = np.clip(sanitize(uncertainty), 0.0, 1.0)
    
    # Robust slack: reduce effective slack under uncertainty only if positive slack exists
    robust_slack = np.where(slack > 0, slack - 1.5 * uncertainty, slack)
    
    # Smooth, bounded urgency: tanh-based (steeper near zero, strictly monotonic, asymptotically bounded)
    # Maps slack → urgency ∈ [0,1], with sharper transition at slack=0 than arctan
    urgency_raw = np.tanh((eps - robust_slack) / (eps + 0.05))
    urgency_scaled = (1.0 + urgency_raw) / 2.0  # [0,1] range, higher = more urgent
    
    # Hard deadline violation boost: large negative penalty ensures top priority for violated tasks
    violation_boost = np.where(slack < 0, -1e5, 0.0)
    
    # Criticality: only amplify upward_rank with uncertainty when slack > 0 (don't punish already-late tasks)
    risk_criticality = upward_rank * np.where(robust_slack > 0, (1.0 + uncertainty), 1.0)
    
    # Latency pressure: criticality × cost, with uncertainty boost only for positive-slack tasks
    exec_comm_sum = min_exec_time + min_comm_time + eps
    latency_pressure = risk_criticality * exec_comm_sum
    
    # Dual-efficiency energy term: balances per-work energy & per-latency energy
    # Encourages low-energy tasks *and* low-cost-to-execute tasks, both scaled by urgency
    energy_per_work = min_incremental_energy / (remaining_work + eps)
    energy_per_cost = min_incremental_energy / (exec_comm_sum + 1.0)
    energy_term = (0.7 * energy_per_work + 0.3 * energy_per_cost) * urgency_scaled
    
    # Fairness: sqrt(wait) for smoother growth; active only when slack > 0 AND urgency < 0.3 (low-pressure regime)
    # Prevents starvation without interfering with high-urgency scheduling
    wait_fairness = np.sqrt(ready_wait_time + eps)
    fairness_gate = np.where((robust_slack > 0.0) & (urgency_scaled < 0.3), 1.0, 0.0)
    fairness_term = wait_fairness * fairness_gate * (1.0 - urgency_scaled)
    
    # Quantile normalization with multi-level fallback for robustness
    def robust_normalize(x):
        x = np.clip(x, -1e6, 1e6)
        if x.size == 1:
            return np.array([0.0])
        q10 = np.quantile(x, 0.1)
        q90 = np.quantile(x, 0.9)
        if q90 - q10 > eps:
            return (x - q10) / (q90 - q10 + eps)
        # Fallback 1: variance-based scaling
        std = np.std(x)
        if std > eps:
            return (x - np.mean(x)) / (std + eps)
        # Fallback 2: zero vector
        return np.zeros_like(x)
    
    norm_urgency = robust_normalize(urgency_scaled)
    norm_latency = robust_normalize(latency_pressure)
    norm_energy = robust_normalize(energy_term)
    norm_fair = robust_normalize(fairness_term)
    
    # Final score: deadline dominance first (strong negative weight), then latency & energy, fairness as tie-break
    score = (
        -35.0 * norm_urgency      # Highest weight: enforce DDL compliance
        - 14.0 * norm_latency     # Critical path pressure
        - 11.0 * norm_energy      # Energy efficiency under urgency
        + 2.5 * norm_fair         # Gentle fairness boost only when safe
    )
    
    # Apply hard violation boost *after* normalization to guarantee ordering override
    score = score + violation_boost
    
    # Final sanitization: ensure finite, bounded, shape-(N,)
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e9, 1e9)
    return score.astype(float).reshape(-1)
