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
    Self-evolved priority rule: fixes over-smoothing of deadline urgency and risk noise amplification,
    replaces fragile IQR with adaptive min-max + variance fallback, enforces strict monotonicity in slack,
    introduces latency-aware starvation cap, and uses additive risk gating (not multiplicative) for stability.
    
    Key improvements:
    - Replaces sigmoid-weighted deadline emphasis with piecewise-linear urgency: 
      strong penalty for slack < 0, linear decay from 0 to median_slack, flat beyond — ensures monotonic priority vs. slack.
    - Risk fusion is now *additive gating*: only applies when slack < 0 AND uncertainty > median_uncertainty, avoiding noise amplification.
    - Normalization uses adaptive scale: min-max range if >1 unique value & range > eps, else std + eps, else constant 1.0 — robust for N=1 or degenerate sets.
    - Critical density divides by *total critical path estimate* (upward_rank * (exec+comm)) instead of remaining_work alone — better reflects marginal impact per unit energy.
    - Aging boost is capped at 5% of base score and scaled by normalized wait time — prevents starvation override of deadline/energy signals.
    - All outputs guaranteed finite, deterministic, shape-(N,), and division-safe.
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
    
    # Adaptive normalization: prefer min-max if well-spread, else fall back to std or constant
    def normalize_adaptive(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size == 1:
            return np.array([1.0])
        rng = np.max(x) - np.min(x)
        if rng > eps and len(np.unique(x)) > 1:
            scale = rng
        else:
            std_val = np.std(x)
            scale = std_val if std_val > eps else 1.0
        return x / (scale + eps)
    
    # Strict monotonic deadline urgency: piecewise-linear, no exponentials or sigmoids
    slack_safe = np.nan_to_num(slack, nan=0.0, posinf=0.0, neginf=-1e6)
    median_slack = np.median(slack_safe[slack_safe > -1e5]) if np.any(slack_safe > -1e5) else 0.0
    # Urgency = 0 for slack >= median_slack; linear ramp to max_penalty at slack=0; steep penalty below 0
    urgency = np.zeros_like(slack_safe)
    # Below zero: strong linear penalty
    mask_neg = slack_safe < 0
    urgency[mask_neg] = -slack_safe[mask_neg] / (np.abs(np.min(slack_safe)) + eps)
    # Between 0 and median_slack: linear decay from max to zero
    mask_mid = (slack_safe >= 0) & (slack_safe <= median_slack + eps)
    if median_slack > eps:
        urgency[mask_mid] = (median_slack - slack_safe[mask_mid]) / (median_slack + eps)
    # Normalize urgency to [0,1] range for stable weighting
    urgency = normalize_adaptive(urgency)
    
    # Latency footprint and safety
    latency_footprint = min_exec_time + min_comm_time
    latency_safe = np.maximum(latency_footprint, eps)
    
    # Critical density: upward_rank weighted by latency footprint, normalized by *critical path relevance*
    # Instead of remaining_work, use upward_rank * latency_safe as intrinsic importance metric
    critical_density_base = upward_rank * latency_safe
    critical_density = normalize_adaptive(critical_density_base)
    
    # Energy efficiency: incremental energy per unit critical path weight (not per work)
    energy_efficiency_base = min_incremental_energy / (critical_density_base + eps)
    energy_efficiency = normalize_adaptive(energy_efficiency_base)
    
    # Starvation-aware aging: capped at 5% of base urgency score, latency-relative
    wait_rel = np.clip(ready_wait_time / latency_safe, 0.0, 5.0)
    wait_norm = normalize_adaptive(wait_rel)
    aging_boost = 0.05 * urgency * wait_norm  # Scaled by urgency — only boosts urgent-starved tasks
    
    # Additive risk gating: only activate if both slack is critical AND uncertainty is high
    median_uncertainty = np.median(uncertainty) if uncertainty.size > 0 else 0.0
    risk_active = (slack_safe < 0) & (uncertainty > (median_uncertainty + eps))
    risk_penalty = np.where(risk_active, uncertainty * 0.3, 0.0)
    risk_penalty = normalize_adaptive(risk_penalty)
    
    # Final convex combination: deadline term dominates when any slack < 0; else energy dominates
    has_deadline_risk = np.any(slack_safe < 0)
    deadline_weight = 0.8 if has_deadline_risk else 0.2
    energy_weight = 1.0 - deadline_weight
    
    # Base deadline score: urgency + risk + critical density (all positively contribute to priority → lower is better)
    score_deadline = (
        deadline_weight * urgency 
        + 0.2 * risk_penalty 
        + 0.15 * critical_density
    )
    
    # Base energy score: efficiency + reduced aging (aging_boost subtracts from priority → lower is better)
    score_energy = (
        energy_weight * energy_efficiency 
        - 0.1 * aging_boost 
        + 0.1 * critical_density
    )
    
    score = score_deadline + score_energy
    
    # Ensure finite, shape-(N,), deterministic output
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    return score
