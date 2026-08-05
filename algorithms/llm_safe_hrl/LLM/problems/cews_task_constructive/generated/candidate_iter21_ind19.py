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
    v5: Hard-DDL-first with *urgency-graded energy efficiency*, *criticality-aware fairness*, 
        *robust uncertainty gating*, and *zero-latency penalty amplification*.
    
    Key evolutions:
    - Urgency now uses *graded penalty tiers*: (slack <= 0) → high-penalty linear; 
      (0 < slack <= 1.0) → medium-penalty quadratic decay; (1.0 < slack <= 3.0) → low-penalty linear fade;
      (slack > 3.0) → zero — sharper deadline enforcement and smoother safety-margin transition.
    - Energy-efficiency term replaced by *inverse ELR boost*: 1/(ELR + eps) when slack > 0, 
      directly promoting low-energy-per-latency tasks in safe regions — more discriminative than masking.
    - Fairness now *criticality-weighted wait pressure*: (ready_wait_time / median(exec_comm_sum + eps)) * sigmoid(upward_rank),
      preventing starvation of high-importance long-waiting tasks without over-prioritizing low-rank ones.
    - Uncertainty gating upgraded to *three-tier risk sensitivity*: 
        (slack <= 0) → full additive boost (high-risk urgency); 
        (0 < slack <= 1.5) → scaled boost (moderate-risk awareness); 
        (slack > 1.5) → zero — eliminates noise amplification in safe regimes.
    - Zero-risk dominance refined: instead of +inf penalty on non-violating tasks, applies *relative urgency penalty* 
      proportional to (min_slack - slack) for all slack > 0 when any violation exists — preserves ranking fidelity while enforcing DDL priority.
    - All min-max normalization robustly handles N=1 via identity fallback (no zero-array), avoids division-by-zero in median, and clips outliers before scaling.
    - Final score strictly finite, deterministic, and reshaped to (N,).
    """
    eps = 1e-08
    N = len(slack)
    
    # Defensive casting and nan/inf handling
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=1e6, neginf=-1e6)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=0.0, posinf=1e6, neginf=0.0)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=0.0, posinf=1e6, neginf=0.0)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=1e6, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=0.0, posinf=1e6, neginf=0.0)
    
    # Graded urgency: piecewise with linear, quadratic, and fade zones
    urgency_raw = np.zeros_like(slack)
    # Violation zone: linear penalty
    mask_viol = slack <= 0
    urgency_raw[mask_viol] = -slack[mask_viol] * 7.0
    # Critical zone: quadratic decay (0 < slack <= 1.0)
    mask_crit = (slack > 0) & (slack <= 1.0)
    urgency_raw[mask_crit] = 7.0 * (1.0 - (slack[mask_crit] ** 2))
    # Caution zone: linear fade (1.0 < slack <= 3.0)
    mask_caut = (slack > 1.0) & (slack <= 3.0)
    urgency_raw[mask_caut] = 7.0 * (1.0 - (slack[mask_caut] - 1.0) / 2.0)
    # Safe zone: zero
    
    # ELR and inverse-ELR boost (energy efficiency under safety)
    exec_comm_sum = min_exec_time + min_comm_time + eps
    elr_base = np.clip(min_incremental_energy / (exec_comm_sum + 3.0), 0.0, 1e5)
    inv_elr_boost = np.where(slack > 0, np.clip(1.0 / (elr_base + eps), 0.0, 1e5), 0.0)
    
    # Slack-gated criticality with sigmoid weighting on upward_rank (preserves rank importance)
    slack_sigmoid = 1.0 / (1.0 + np.exp(-(slack - 0.5) / (0.5 + eps)))
    criticality_raw = upward_rank * remaining_work * slack_sigmoid
    
    # Criticality-weighted wait pressure (fairness that respects importance)
    exec_comm_median = np.median(exec_comm_sum) if N > 1 else exec_comm_sum[0]
    wait_pressure = np.clip(ready_wait_time / (exec_comm_median + eps), 0.0, 10.0)
    rank_sigmoid = 1.0 / (1.0 + np.exp(-(upward_rank - np.median(upward_rank + eps)) / (np.std(upward_rank) + eps)))
    fairness_raw = wait_pressure * rank_sigmoid * np.clip(np.maximum(0.0, slack) / (3.0 + eps), 0.0, 1.0)
    
    # Three-tier uncertainty gating: full → scaled → zero
    unc_boost = np.zeros_like(uncertainty)
    mask_unc_high = slack <= 0
    unc_boost[mask_unc_high] = np.clip(uncertainty[mask_unc_high] * 0.5, 0.0, 0.25)
    mask_unc_med = (slack > 0) & (slack <= 1.5)
    unc_boost[mask_unc_med] = np.clip(uncertainty[mask_unc_med] * 0.25, 0.0, 0.12)
    
    # Zero-risk dominance: relative urgency penalty for non-violating tasks when violations exist
    has_violation = np.any(slack <= 0)
    min_slack = np.min(slack)
    rel_urgency_penalty = np.where(has_violation & (slack > 0), (slack - min_slack) * 5.0, 0.0)
    
    # Robust min-max normalization (handles N=1 safely)
    def normalize_minmax(x):
        x = np.clip(x, -1e6, 1e6)
        if x.size == 1:
            return np.array([0.0])
        x_min, x_max = np.min(x), np.max(x)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        return (x - x_min) / (x_max - x_min + eps)
    
    norm_urgency = normalize_minmax(urgency_raw)
    norm_inv_elr = normalize_minmax(inv_elr_boost)
    norm_criticality = normalize_minmax(criticality_raw)
    norm_fairness = normalize_minmax(fairness_raw)
    norm_unc = normalize_minmax(unc_boost)
    norm_rel_penalty = normalize_minmax(rel_urgency_penalty)
    
    # Weighted terms: urgency and energy efficiency dominate; criticality & fairness moderate; uncertainty & penalty refine
    urgency_term = -9.0 * norm_urgency
    inv_elr_term = -5.0 * norm_inv_elr
    criticality_term = 0.7 * norm_criticality
    fairness_term = -0.45 * norm_fairness
    unc_term = -0.18 * norm_unc
    risk_penalty_term = 0.8 * norm_rel_penalty
    
    score = (
        urgency_term + 
        inv_elr_term + 
        criticality_term + 
        fairness_term + 
        unc_term + 
        risk_penalty_term
    )
    
    # Final safeguard: finite, deterministic, shape-correct
    score = np.nan_to_num(score, nan=1e8, posinf=1e8, neginf=-1e8)
    score = np.clip(score, -1e8, 1e8)
    return score.reshape(-1)
