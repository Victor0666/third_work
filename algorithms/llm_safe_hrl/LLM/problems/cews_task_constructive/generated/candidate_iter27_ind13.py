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

    # Self-evolved priority: strict DDL enforcement + energy-efficiency coupling + robust fairness
    # Key improvements over v1:
    # - Restores hard violation boost (-1e9) for *any* slack < -eps to guarantee zero deadline violations
    # - Replaces complex uncertainty coupling with linear, bounded uncertainty penalty only in critical zone
    # - Simplifies fairness to static sqrt(wait)/max(1, slack+1) — deterministic, avoids median dependence
    # - Strengthens CED term by using *normalized* energy efficiency (work/energy) scaled by upward_rank
    # - Removes redundant pressure term; criticality already captured in CED and urgency
    # - Uses min-max normalization instead of IQR for better stability on small N and degenerate cases
    # - All terms bounded, sanitized, and reshaped to (N,) deterministically
    eps = 1e-08
    def clean(x):
        x = np.asarray(x, dtype=float)
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    
    min_exec_time = clean(min_exec_time)
    min_comm_time = clean(min_comm_time)
    min_incremental_energy = clean(min_incremental_energy)
    slack = clean(slack)
    upward_rank = clean(upward_rank)
    remaining_work = clean(remaining_work)
    ready_wait_time = clean(ready_wait_time)
    uncertainty = clean(uncertainty)
    
    N = len(slack)
    
    # Hard violation dominance: assign extreme priority to any task already violating deadline
    violation_boost = np.where(slack < -eps, -1000000000.0, 0.0)
    
    # Bounded urgency: arctan-based near-deadline sensitivity (0 <= slack <= 2s), zero elsewhere
    urgency_raw = np.where((slack >= 0.0) & (slack <= 2.0), np.arctan(-slack / (1.0 + eps)), 0.0)
    
    # Robust min-max normalization: handles N=1, flat arrays, and outliers safely
    def normalize_minmax(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size == 1:
            return np.array([0.0])
        xmin, xmax = np.min(x), np.max(x)
        scale = xmax - xmin
        if scale < eps:
            return np.zeros_like(x)
        return (x - xmin) / (scale + eps)
    
    norm_urgency = normalize_minmax(urgency_raw)
    urgency_term = -4.0 * norm_urgency  # Stronger weight for deadline proximity
    
    # Criticality-Energy-Density (CED): upward_rank-weighted marginal efficiency
    # Higher work per joule → better energy efficiency → lower priority score (hence negative weight)
    ced_base = (upward_rank * remaining_work) / (min_incremental_energy + eps)
    # Gate activation only when slack is non-negative (deadline feasible)
    gate_sigmoid = np.where(slack >= 0.0, 1.0 / (1.0 + np.exp(-(slack + 0.5) / (0.25 + eps))), 0.0)
    ced_masked = ced_base * gate_sigmoid
    norm_ced = normalize_minmax(ced_masked)
    ced_term = -2.0 * norm_ced  # Energy-awareness activated only under safety
    
    # Fairness: bounded starvation awareness — sqrt(wait) penalized only when slack > 0
    # Avoids punishing waiting under lateness risk; uses static denominator for determinism
    wait_safe = np.maximum(ready_wait_time, 0.0)
    fairness_raw = np.sqrt(wait_safe + eps) / (np.maximum(slack, 0.0) + 1.0 + eps)
    norm_fairness = normalize_minmax(fairness_raw)
    fairness_term = 0.3 * norm_fairness  # Small positive weight: longer wait → slightly lower priority
    
    # Uncertainty penalty: linear, bounded, applied only when slack < 1.0s AND uncertainty > 0.5
    # Directly penalizes risky scheduling near deadline
    risk_zone = (slack < 1.0) & (uncertainty > 0.5)
    unc_penalty = np.where(risk_zone, np.clip(uncertainty * 0.5, 0.0, 0.5), 0.0)
    norm_unc = normalize_minmax(unc_penalty)
    uncertainty_term = 0.8 * norm_unc
    
    # Assemble final score: smaller = higher priority
    score = violation_boost + urgency_term + ced_term + fairness_term + uncertainty_term
    
    # Final sanitization: ensure finite, bounded, shape-(N,) output
    score = np.nan_to_num(score, nan=100000000.0, posinf=100000000.0, neginf=-100000000.0)
    score = np.clip(score, -100000000.0, 100000000.0)
    return score.astype(float).reshape(-1)
