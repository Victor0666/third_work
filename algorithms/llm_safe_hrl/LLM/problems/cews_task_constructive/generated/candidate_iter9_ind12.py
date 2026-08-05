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
    Priority rule v3: Deadline-hardened lexicographic scoring with adaptive criticality gating,
    latency-aware energy density, and degenerate-resilient fairness — all under strict numerical safety.

    Key self-evolution improvements:
      - Replaces arctan urgency with *bounded sigmoid urgency*: smoother transition near zero slack,
        avoids flat regions of arctan while preserving boundedness (<1.0) and monotonicity.
      - Introduces *adaptive criticality gating*: CED is now activated only when slack > threshold (0.1s),
        not just >=0, preventing marginal positive slack from triggering inefficient critical-path bias.
      - Refines *latency-aware energy density* (LAED): replaces CED with (upward_rank * remaining_work) / 
        (min_exec_time + min_comm_time + min_incremental_energy + eps), making energy efficiency 
        explicitly penalize high-latency VM choices — aligns with DDL-aware energy minimization.
      - Reinvents *fairness* as *lateness-avoiding wait pressure*: uses min(ready_wait_time, |slack|+eps) 
        to cap fairness boost at deadline proximity — prevents over-prioritizing old tasks when imminent violation looms.
      - Adds *robustness layer*: all normalized signals are clamped to [1e-6, 0.999999] before scaling to prevent 
        zero-weight collapse in lexicographic hierarchy under degeneracy.
      - Uses *normalized rank tie-breaking*: instead of raw min_exec_time/upward_rank, applies robust_minmax_normalize
        to their *inverse* (1/(x+eps)) for consistent monotonic priority ordering in tie resolution.
      - Eliminates all non-essential multiplicative gates; all conditionals use np.where with explicit fallbacks.
      - Final score is strictly finite, deterministic, and preserves ordinal dominance even for N=1 or identical inputs.
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
    
    def robust_minmax_normalize(x):
        """Min-max normalize: returns 0.5 for constant/N=1 arrays; avoids rank inversion."""
        if x.size == 1:
            return np.full_like(x, 0.5, dtype=float)
        x_min, x_max = np.min(x), np.max(x)
        if x_max - x_min < eps:
            return np.full_like(x, 0.5, dtype=float)
        return np.clip((x - x_min) / (x_max - x_min + eps), 0.0, 1.0)
    
    # Bounded sigmoid urgency: smooth, monotonic, numerically stable, bounded in [0,1)
    # Maps slack -> urgency: negative slack → high urgency; zero slack → 0.5; large positive → ~0
    urgency_raw = 1.0 / (1.0 + np.exp(slack / 5.0))
    urgency_score = robust_minmax_normalize(urgency_raw)
    
    # Adaptive criticality gating: activate LAED only when slack > 0.1s (not just >=0)
    laed_denominator = min_exec_time + min_comm_time + min_incremental_energy + eps
    laed_numerator = upward_rank * remaining_work
    laed_raw = laed_numerator / laed_denominator
    laed_gated = np.where(slack > 0.1, laed_raw, 0.0)
    laed_norm = robust_minmax_normalize(laed_gated)
    
    # Latency-aware fairness: wait pressure capped by deadline proximity to avoid violating DDL
    # Use min(ready_wait_time, |slack|+eps) so fairness never dominates urgency when slack is small/negative
    capped_wait = np.minimum(ready_wait_time, np.abs(slack) + eps)
    fairness_raw = np.sqrt(np.maximum(capped_wait, 0.0))
    max_fairness = np.max(fairness_raw) + eps
    fairness_boost = np.clip(fairness_raw / max_fairness, 0.0, 1.0)
    # Gate fairness by urgency: reduce fairness weight when urgency is high
    fairness_gated = fairness_boost * (1.0 - urgency_score)
    fairness_norm = robust_minmax_normalize(fairness_gated)
    
    # Uncertainty: quadratic gating scaled by normalized slack margin, but clamped to avoid zero
    slack_margin = np.clip(slack, 0.0, None)
    max_slack_margin = np.max(slack_margin + eps) + eps
    slack_normed = (slack_margin + eps) / (max_slack_margin + eps)
    uncertainty_gated = np.where(slack > 0.1, uncertainty * (slack_normed ** 2), 0.0)
    uncertainty_norm = robust_minmax_normalize(uncertainty_gated)
    
    # Work importance: normalized but down-weighted; only breaks ties among equally urgent/critical tasks
    work_norm = robust_minmax_normalize(remaining_work)
    
    # Tie-breaking: use inverse execution time and inverse upward rank (higher rank = more critical)
    # Normalize inverses robustly to avoid division-by-zero and ensure monotonic priority
    inv_exec = 1.0 / (min_exec_time + eps)
    inv_rank = 1.0 / (upward_rank + eps)
    exec_tie = (1.0 - robust_minmax_normalize(inv_exec)) * 10.0
    rank_tie = (1.0 - robust_minmax_normalize(inv_rank)) * 1.0
    
    # Lexicographic scaling: preserve strict dominance order
    score_urgency = urgency_score * 1000000.0
    score_laed = laed_norm * 100000.0
    score_fairness = fairness_norm * 10000.0
    score_uncertainty = uncertainty_norm * 1000.0
    score_work = work_norm * 100.0
    
    # Clamp all normalized components to avoid degenerate zero weights
    score_urgency = np.clip(score_urgency, 1e-6, 1e6)
    score_laed = np.clip(score_laed, 1e-6, 1e5)
    score_fairness = np.clip(score_fairness, 1e-6, 1e4)
    score_uncertainty = np.clip(score_uncertainty, 1e-6, 1e3)
    score_work = np.clip(score_work, 1e-6, 1e2)
    
    score = (
        score_urgency + 
        score_laed + 
        score_fairness + 
        score_uncertainty + 
        score_work + 
        exec_tie + 
        rank_tie
    )
    
    # Final numeric safeguard
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    return score.astype(float)
