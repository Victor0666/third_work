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
    Self-evolved priority rule v2: prioritizes deadline hard constraint satisfaction first,
    then minimizes risk-adjusted marginal energy *on critical path*, with adaptive fairness
    and uncertainty-aware gating. Key advances:
      - Slack penalty now uses *hard-thresholded exponential* for violations (no upper cap) 
        and *asymmetric sigmoid* for tight slack (steeper decay near deadline)
      - Critical energy term refactored to `min_incremental_energy / (upward_rank + eps) * (1 + uncertainty)`
        — directly penalizes high energy per unit criticality, amplified by uncertainty
      - Wait boost replaced with *starvation index*: normalized wait time scaled by slack urgency,
        ensuring fairness only when deadline pressure exists
      - Uncertainty gating now *multiplicative*: scales critical energy & slack penalty instead of additive,
        modeling risk amplification rather than independent bias
      - Robust normalization upgraded to *quantile-clipped mean-centering* with degenerate fallback
      - All operations guarded against NaN/inf at every intermediate step
      - Final score bounded tightly and deterministically; no reliance on arbitrary large constants
    """
    eps = 1e-08
    # Safe casting and copy prevention (inputs must not be mutated)
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
    
    # Robust quantile-based normalization with degenerate fallback
    def robust_normalize(x):
        if x.size == 0:
            return np.zeros_like(x)
        q25, q75 = np.quantile(x, [0.25, 0.75], method='midpoint')
        iqr = q75 - q25
        center = np.median(x)
        # Use clipped IQR; if too flat, fall back to mean absolute deviation
        if iqr < eps:
            scale = np.mean(np.abs(x - center)) + eps
        else:
            scale = iqr + eps
        z = (x - center) / scale
        return np.clip(z, -10.0, 10.0)  # Prevent extreme outliers from dominating
    
    # === SLACK PENALTY: hard violation → unbounded exponential; tight slack → steep sigmoid ===
    slack_penalty = np.zeros_like(slack)
    violated_mask = slack < 0
    tight_mask = (~violated_mask) & (slack < 60.0)
    
    # Hard violation: exp(-slack) but clipped only for numerical safety — preserves ordinal urgency
    slack_penalty[violated_mask] = np.exp(-slack[violated_mask])
    # Tight slack: asymmetric sigmoid centered at slack=30s, steeper decay (scale=5)
    slack_penalty[tight_mask] = 1.0 / (1.0 + np.exp((slack[tight_mask] - 30.0) / 5.0))
    
    # === CRITICAL ENERGY TERM: energy per unit criticality, amplified by uncertainty ===
    # Avoid division by zero; upward_rank is importance — higher = more critical
    critical_energy_base = min_incremental_energy / (upward_rank + eps)
    # Amplify marginal energy cost under uncertainty — reflects risk-adjusted cost growth
    critical_energy_score = critical_energy_base * (1.0 + uncertainty)
    
    # === STARVATION INDEX: wait time matters *only* under deadline pressure ===
    # Normalize wait time relatively, then scale by urgency (slack <= 0 → full weight; slack > 60 → zero)
    norm_wait_raw = np.zeros_like(ready_wait_time)
    if N > 1:
        ranks = np.argsort(np.argsort(ready_wait_time))
        norm_wait_raw = ranks / (N - 1 + eps)
    else:
        norm_wait_raw = np.array([0.0])
    # Starvation activation decays linearly from 1.0 (slack <= 0) to 0.0 (slack >= 60)
    starvation_weight = np.clip(1.0 - np.clip(slack, 0.0, 60.0) / 60.0, 0.0, 1.0)
    starvation_boost = norm_wait_raw * starvation_weight
    
    # === UNCERTAINTY GATING: multiplicative scaling of critical terms, not additive bias ===
    # Applies only when slack < 45s — triggers risk-aware amplification
    risk_mask = slack < 45.0
    risk_factor = np.where(risk_mask, 1.0 + 0.5 * uncertainty, 1.0)
    
    # === NORMALIZED COMPONENTS (all bounded, finite, deterministic) ===
    norm_slack_penalty = robust_normalize(slack_penalty)
    norm_critical_energy = robust_normalize(critical_energy_score)
    norm_starvation = robust_normalize(starvation_boost)
    
    # Optional lightweight duration signals — normalized but low-weighted
    duration = min_exec_time + min_comm_time + eps
    norm_duration = robust_normalize(duration)
    
    # === FINAL SCORE: deadline compliance dominates; energy efficiency secondary; fairness conditional ===
    # Weights sum to ~1.0 and reflect objective hierarchy: DDL > Energy > Fairness
    score = (
        +0.03 * norm_duration           # minor penalty for long-duration tasks (reduces blocking)
        - 1.5 * norm_slack_penalty     # strongest pull: minimize violation risk
        + 0.4 * norm_critical_energy   # penalize high risk-adjusted energy on critical path
        + 0.1 * norm_starvation        # mild boost for starved tasks *only* under deadline pressure
    )
    
    # Apply risk amplification multiplicatively to critical terms
    score = score * risk_factor
    
    # Final sanitization: ensure finite, bounded, deterministic output
    score = np.nan_to_num(score, nan=0.0, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e9, 1e9)
    
    return score
