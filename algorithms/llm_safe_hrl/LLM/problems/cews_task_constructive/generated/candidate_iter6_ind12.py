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
    Self-evolved priority rule: fixes slack=0 discontinuity, restores uncertainty amplification for tight-but-positive slack,
    introduces slack-aware aging decay, and unifies deadline modeling with monotonic smoothness + bounded hard penalty.
    
    Key improvements:
    - Smooth deadline risk: arctan(-slack) → full-range monotonic urgency (no clipping), scaled to [0,1]
    - Hard penalty only for slack < -eps, applied *additively* but bounded to avoid dominance over energy efficiency
    - Uncertainty gated by (0 < slack <= median_slack) — targets high-risk "tight-margin" tasks where energy optimization matters most
    - Aging boost scaled by sigmoid(1 - normalized_slack) → decays smoothly as deadline pressure increases, preventing starvation without violating DDL
    - Critical-energy density retained as primary efficiency signal, now weighted higher under positive-slack regimes
    - All normalizations use robust MAD; degenerate cases (N=1, constant) handled explicitly
    - No double-counting: removed redundant remaining_work in deadline term; upward_rank already captures criticality
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
    
    def safe_mad_normalize(x):
        """Robust MAD normalization: handles N=1, constant arrays, NaN/inf."""
        if x.size == 0:
            return np.zeros(0)
        x_clean = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x_clean.size == 1:
            return np.zeros_like(x_clean)
        median_x = np.median(x_clean)
        mad = np.median(np.abs(x_clean - median_x)) + eps
        if mad < eps:
            return np.zeros_like(x_clean)
        normed = (x_clean - median_x) / mad
        return np.clip(normed, -3.0, 3.0)
    
    # Full-range smooth deadline urgency: arctan(-slack) maps (-∞,∞) → (0,π), then normalized to [0,1]
    deadline_urgency = (np.arctan(-slack) + np.pi/2) / np.pi  # [0,1], monotonic: higher when slack ↓
    # Bounded hard penalty only for violated deadlines (slack < -eps), capped at 0.3 to preserve energy tradeoff
    hard_penalty = np.where(slack < -eps, np.clip(-slack / (np.abs(np.min(slack)) + eps), 0.0, 0.3), 0.0)
    deadline_risk = deadline_urgency + hard_penalty
    deadline_score = safe_mad_normalize(deadline_risk)
    
    # Critical-energy density: prioritize high DAG importance per joule (efficiency under deadline feasibility)
    critical_energy_density = upward_rank * remaining_work / (min_incremental_energy + eps)
    critical_energy_norm = safe_mad_normalize(critical_energy_density)
    
    # Upward rank alone — structural importance, normalized separately
    upward_rank_norm = safe_mad_normalize(upward_rank)
    
    # Slack-aware aging boost: decays via sigmoid as slack decreases → prevents starvation only when safe
    slack_clean = np.clip(slack, -1e6, 1e6)  # avoid overflow in exp
    slack_norm_for_decay = (slack_clean - np.median(slack_clean)) / (np.median(np.abs(slack_clean - np.median(slack_clean))) + eps)
    # Sigmoid decay: ~1.0 when slack >> 0, ~0.0 when slack ≤ 0
    aging_decay_factor = 1.0 / (1.0 + np.exp(-(slack_norm_for_decay - 0.5)))
    wait_normalized = np.clip(ready_wait_time / (np.max(min_exec_time + min_comm_time + eps) + eps), 0.0, 1.0)
    aging_boost = 0.05 * wait_normalized * aging_decay_factor
    
    # Uncertainty gating: active only for *tight but feasible* slack — i.e., 0 < slack ≤ median_slack
    # This aligns with DDL-aware energy optimization: uncertainty matters most where margin is thin but non-zero
    median_slack = np.median(slack[slack > 0]) if np.any(slack > 0) else np.median(np.abs(slack)) + eps
    uncertainty_active = (slack > 0) & (slack <= median_slack + eps)
    uncertainty_gated = np.where(uncertainty_active, uncertainty, 0.0)
    uncertainty_norm = safe_mad_normalize(uncertainty_gated)
    
    # Weight hierarchy: deadline dominance preserved (4.0), efficiency boosted under slack (2.4), aging & uncertainty moderated
    score = (
        +4.0 * deadline_score 
        - 2.4 * critical_energy_norm 
        - 1.0 * upward_rank_norm 
        + 0.05 * aging_boost 
        + 0.22 * uncertainty_norm
    )
    
    # Final sanitization: ensure finite, shape-(N,) output
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    return score
