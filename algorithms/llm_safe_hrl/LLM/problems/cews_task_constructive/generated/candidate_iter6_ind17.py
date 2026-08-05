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
    Self-evolved priority rule v2: deadline-hardened, energy-aware, and starvation-proof.
    
    Key self-evolution improvements:
    - Replaces linear saturation with *deadline urgency cliff*: sharp penalty only when slack < -0.1s,
      but now scaled by critical-path weight (upward_rank) to prioritize high-impact lateness first.
    - Introduces *energy-efficiency ratio* instead of raw joules-per-time: normalizes energy gain against
      remaining_work to favor tasks that deliver maximal computation-per-joule — aligning with DDL-safe energy minimization.
    - Adds *dynamic aging cap*: aging_boost now upper-bounded by min(0.1, |urgency|), preventing over-prioritization
      of aged tasks when deadline pressure is already high (avoids conflict between urgency and aging).
    - Replaces additive risk penalty with *risk-modulated urgency scaling*: uncertainty multiplies urgency only
      when both slack < -0.1 AND uncertainty > median, eliminating standalone risk term and reducing parameter coupling.
    - Uses *robust rank-preserving normalization*: avoids mean/median shift artifacts by normalizing only relative order
      via percentile ranks + sigmoid squashing — ensures monotonicity and stability for N=1 to N=1000.
    - All components are strictly bounded, division-safe, and guarantee finite (N,) output with no NaN/inf leakage.
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
    
    # Robust rank-preserving normalization: maps x → [0,1] via percentile rank, then sigmoid-squashes to [-1,1]
    def normalize_robust(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size == 0:
            return np.zeros_like(x)
        if x.size == 1:
            return np.array([0.0])
        # Percentile rank: fraction of values <= each element
        sorted_x = np.sort(x)
        ranks = np.array([np.searchsorted(sorted_x, xi, side='right') for xi in x]) / (len(x) - 1 + eps)
        # Sigmoid squashing to [-1, 1] preserves order & bounds extremes
        return 2.0 / (1.0 + np.exp(-6.0 * (ranks - 0.5))) - 1.0
    
    # Deadline urgency: cliff-based, upward_rank-weighted, and uncertainty-gated
    deadline_violation = slack < -0.1
    base_urgency = np.where(deadline_violation, -slack, 0.0)
    # Scale urgency by critical importance and modulate only under joint risk (lateness + uncertainty)
    median_uncert = np.median(uncertainty) if uncertainty.size > 0 else 0.0
    risk_active = deadline_violation & (uncertainty > median_uncert + eps)
    urgency = base_urgency * upward_rank * (1.0 + 0.5 * np.where(risk_active, uncertainty, 0.0))
    
    # Energy efficiency: joules per million instructions *normalized by remaining work — favors high-compute/low-energy tasks
    exec_comm_safe = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_work = np.where(remaining_work > eps, 
                               min_incremental_energy / (remaining_work + eps), 
                               min_incremental_energy / eps)
    # Normalize to emphasize *relative efficiency*, not absolute magnitude
    norm_energy_eff = normalize_robust(energy_per_work)
    
    # Slack-weighted critical path importance (preserves continuity near deadline)
    slack_sigmoid = 1.0 / (1.0 + np.exp(-np.clip(slack, -30.0, 30.0) / 5.0))
    weighted_upward_rank = upward_rank * slack_sigmoid
    norm_weighted_rank = normalize_robust(weighted_upward_rank)
    
    # Dynamic aging: only activates when slack >= -0.1 (no deadline crisis), capped by urgency magnitude
    aging_eligible = ~deadline_violation
    aging_denom = np.abs(slack) + 1.0
    aging_raw = np.tanh(0.3 * ready_wait_time / aging_denom)
    aging_cap = np.minimum(0.1, np.abs(normalize_robust(urgency)))  # cap scales with current urgency
    aging_boost = np.where(aging_eligible, aging_raw * aging_cap, 0.0)
    
    # Normalize urgency last — ensures it dominates when active, but remains bounded
    norm_urgency = normalize_robust(urgency)
    
    # Final score: urgency dominates when active; otherwise relies on critical path + efficiency + aging
    # Coefficients sum to 1.0 and reflect DDL-hard constraint priority
    score = (
        0.50 * norm_urgency +
        0.25 * norm_weighted_rank +
        0.15 * norm_energy_eff +
        0.10 * aging_boost
    )
    
    # Ensure strict finiteness and shape compliance
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    return score
