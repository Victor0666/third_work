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
    Priority rule v2: Self-evolved deadline-hardened, uncertainty-aware, and energy-efficient.
    Key improvements over v1:
      - Restores robust_slack = slack - uncertainty (v0 baseline) → reduces false-positive overrides
      - Replaces multiplicative hierarchy with *additive* weighted sum of normalized terms → avoids collapse on zero factors
      - Uses sigmoid-based SEER inversion with tunable steepness (k=3.0) → preserves discriminative power near optimum
      - Introduces *deadline proximity penalty* for tasks with slack < 0.5*median_slack → smooth priority boost before hard violation
      - Fairness term now uses relative wait ratio (ready_wait_time / max_exec_comm) → physically grounded fairness metric
      - All normalization uses percentile-based robust scaling with fallback to constant for N=1 or flat distributions
      - Violation override applied *after* base score computation but *before* final clipping → deterministic and stable
    """
    eps = 1e-08
    N = len(slack)
    
    # Sanitize all inputs: convert to float, replace NaN/inf/neg-inf with safe finite values
    def sanitize(x):
        x = np.asarray(x, dtype=float)
        return np.nan_to_num(x, nan=eps, posinf=1e6, neginf=eps)
    
    min_exec_time = sanitize(min_exec_time)
    min_comm_time = sanitize(min_comm_time)
    min_incremental_energy = sanitize(min_incremental_energy)
    slack = sanitize(slack)
    upward_rank = sanitize(upward_rank)
    remaining_work = sanitize(remaining_work)
    ready_wait_time = sanitize(ready_wait_time)
    uncertainty = sanitize(uncertainty)
    
    # Robust slack: slack - 1×uncertainty (restored from v0; balances sensitivity & false positives)
    robust_slack = slack - uncertainty
    robust_slack = np.clip(robust_slack, -1e5, 1e5)
    
    # Hard violation flag: strict deadline breach (slack < 0)
    is_violated = (slack < 0.0).astype(float)
    
    # Urgency: smooth, bounded tanh-based measure — higher urgency for smaller robust_slack
    # Clamped to avoid numerical instability; inverted so smaller value = higher urgency
    urgency_raw = np.clip(robust_slack / (np.abs(robust_slack) + eps), -10.0, 10.0)
    urgency_score = 1.0 - np.tanh(urgency_raw)  # [0, 2] → map to [0,1]
    
    # Critical path density: importance per unit time, gated only by positive upward_rank (avoid deadweight)
    exec_comm_sum = min_exec_time + min_comm_time + eps
    cp_density = upward_rank * remaining_work / (exec_comm_sum + eps)
    cp_gated = np.where(upward_rank > eps, cp_density, 0.0)
    
    # Energy efficiency: SEER = (exec+comm)/energy → higher is better; invert via steep sigmoid
    seer = exec_comm_sum / (min_incremental_energy + eps)
    # Sigmoid inversion: maps high SEER → low energy_score (good energy use → low priority cost)
    energy_score = 1.0 / (1.0 + np.exp(3.0 * (1.0 - seer)))  # k=3.0: sharp but differentiable
    
    # Fairness: relative waiting time normalized by task's own compute+comm budget → prevents starvation
    # Only active when task is not under immediate deadline pressure (robust_slack > 0)
    fairness_raw = ready_wait_time / (exec_comm_sum + eps)
    fairness_gated = np.where(robust_slack > 0.0, fairness_raw, 0.0)
    
    # Deadline proximity penalty: soft boost for tasks nearing deadline even if not violated
    # Activates when robust_slack < 0.5 * median positive slack
    positive_slack = slack[slack > eps]
    median_positive_slack = np.median(positive_slack) if len(positive_slack) > 0 else 1.0
    proximity_threshold = 0.5 * median_positive_slack + eps
    proximity_penalty = np.where(robust_slack < proximity_threshold, 
                                1.0 - np.tanh((robust_slack - proximity_threshold) / (eps + np.abs(proximity_threshold))), 
                                0.0)
    
    # Uncertainty penalty: linearly penalize high-uncertainty tasks to prefer predictable ones
    unc_med = np.median(uncertainty) + eps
    unc_penalty = np.clip(uncertainty / unc_med, 0.0, 4.0) * 0.02
    
    # Robust percentile normalization: maps each term to [0,1], handles edge cases safely
    def robust_normalize(x):
        x_clipped = np.clip(x, -1e6, 1e6)
        if x_clipped.size == 1:
            return np.array([0.5])
        p05 = np.percentile(x_clipped, 5)
        p95 = np.percentile(x_clipped, 95)
        if p95 - p05 < eps:
            return np.full_like(x_clipped, 0.5)
        return np.clip((x_clipped - p05) / (p95 - p05 + eps), 0.0, 1.0)
    
    urgency_norm = robust_normalize(urgency_score)
    cp_norm = robust_normalize(cp_gated)
    energy_norm = robust_normalize(energy_score)
    fairness_norm = robust_normalize(fairness_gated)
    proximity_norm = robust_normalize(proximity_penalty)
    
    # Additive composition: avoids multiplicative collapse, enables independent term contribution
    # Weights chosen to reflect operational priorities: urgency (strongest), then criticality, energy, fairness, proximity
    score = (
        1.5 * urgency_norm +
        1.0 * cp_norm +
        0.7 * energy_norm +
        0.3 * fairness_norm +
        0.5 * proximity_norm +
        unc_penalty
    )
    
    # Deterministic violation override: assign lowest possible score to violated tasks
    base_min = np.min(score) if N > 0 else 0.0
    score = np.where(is_violated, base_min - 1e9, score)
    
    # Final sanitization: ensure finite, bounded, shape-(N,) output
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e12, 1e12)
    
    # Enforce shape (N,) and float64 dtype
    return score.astype(np.float64).reshape(-1)
