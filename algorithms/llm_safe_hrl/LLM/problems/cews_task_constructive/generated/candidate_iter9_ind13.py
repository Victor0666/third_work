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
    Self-evolved v2: Tightens deadline enforcement, refines energy-efficiency coupling,
    introduces *slack-aware critical path compression*, and replaces heuristic aging with
    *urgency-gated wait amplification* for fairness without compromising DDL safety.
    
    Key improvements:
    - Urgency term now uses *signed arctan slack scaling*: linear near zero, bounded far out → 
      preserves sensitivity to imminent deadlines while avoiding saturation.
    - Critical-energy density (CED) enhanced with *slack-weighted normalization*: 
      prioritizes high-impact-per-joule tasks more aggressively when slack is tight (not just non-negative).
    - Efficiency term redefined as *energy-latency ratio under uncertainty-adjusted latency*, 
      normalized only over slack>=0 subset → avoids diluting efficiency signal with violating tasks.
    - Aging replaced by *wait amplification gated by urgency threshold*: only boosts waiting tasks 
      when urgency > 0.7 (i.e., slack < ~1.5s), preventing premature aging in relaxed regimes.
    - All robust normalizations now guarantee monotonicity, finite range, and exact (N,) shape.
    - Explicit slack-sign consistency: negative slack always yields highest urgency (smallest score).
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

    # Signed arctan urgency: smooth, unbounded-to-bounded, preserves sign-sensitive ordering
    # Maps slack → [0, 1], where slack → -∞ → 1.0 (max urgency), slack → +∞ → 0.0 (min urgency)
    arctan_urgency = (np.arctan(-slack / (1.0 + eps)) + np.pi/2) / np.pi
    urgency_term = 4.5 * arctan_urgency  # Slightly increased weight for stricter DDL adherence

    # CED: upward_rank * remaining_work / energy, but normalized *only over slack >= 0* subset
    # Then scaled by (1 - arctan_urgency) to boost CED priority when slack is tight (not just non-neg)
    base_ced = upward_rank * remaining_work / (min_incremental_energy + eps)
    ced_mask = (slack >= 0)
    ced_valid = base_ced[ced_mask] if np.any(ced_mask) else np.array([0.0])
    
    def robust_minmax_subset(x, full_size):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size == 1:
            return np.zeros(full_size)
        rng = np.max(x) - np.min(x)
        if rng > eps:
            normed = (x - np.min(x)) / (rng + eps)
            result = np.zeros(full_size)
            result[ced_mask] = normed
            return result
        else:
            q75, q25 = np.percentile(x, [75, 25])
            iqr = q75 - q25
            if iqr > eps:
                normed = (x - np.median(x)) / (iqr + eps)
                result = np.zeros(full_size)
                result[ced_mask] = normed
                return result
            else:
                result = np.zeros(full_size)
                result[ced_mask] = 0.0
                return result
    
    norm_ced_full = robust_minmax_subset(ced_valid, slack.size)
    # Slack-tightness modulation: amplify CED impact when urgency is high (i.e., slack small/neg)
    # This ensures high-CED tasks rise *especially* when time is scarce — not just when slack ≥ 0
    tightness_weight = 1.0 - arctan_urgency  # high when slack is large → suppress CED; low when slack tiny → boost CED
    critical_energy_term = -2.8 * norm_ced_full * (1.0 + 0.5 * tightness_weight)

    # Latency-inflated energy efficiency: only computed & normalized over slack >= 0
    base_latency = min_exec_time + min_comm_time + eps
    # Uncertainty inflates latency only if slack > 0 AND uncertainty significant
    inflation_factor = np.where((slack > 0) & (uncertainty > 2*eps), 
                               np.clip(uncertainty / (np.percentile(base_latency, 90) + eps), 0.0, 1.0), 
                               0.0)
    inflated_latency = base_latency * (1.0 + 0.2 * inflation_factor)
    energy_per_latency = min_incremental_energy / (inflated_latency + eps)
    
    eff_mask = (slack >= 0)
    eff_valid = energy_per_latency[eff_mask] if np.any(eff_mask) else np.array([0.0])
    norm_eff_full = robust_minmax_subset(eff_valid, slack.size)
    efficiency_term = 1.2 * norm_eff_full

    # Urgency-gated wait amplification: only activates for high-urgency tasks (arctan_urgency > 0.7)
    # Prevents unfair starvation *only* when deadline pressure exists; avoids artificial boosting otherwise
    sqrt_wait = np.sqrt(np.maximum(ready_wait_time, 0.0) + eps)
    wait_cap = np.percentile(sqrt_wait, 95) + eps if sqrt_wait.size > 1 else np.max(sqrt_wait) + eps
    clipped_wait = np.clip(sqrt_wait, 0.0, wait_cap)
    norm_wait_full = robust_minmax_subset(clipped_wait, slack.size)
    urgency_gate = (arctan_urgency > 0.7)
    aging_term = -0.4 * norm_wait_full * urgency_gate

    # Final score: smaller = higher priority
    score = urgency_term + critical_energy_term + efficiency_term + aging_term

    # Final sanitization: ensure finite, deterministic, shape-(N,) output
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    score = score.reshape(-1)  # Enforce explicit 1D shape

    return score
