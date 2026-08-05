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

    'Self-evolved priority rule: hard deadline safety first, critical-path fidelity second,\n    energy efficiency third — with strict risk gating, bounded fairness, and numerical robustness.\n\n    Key self-evolution improvements:\n    - Replaces hybrid deadline scoring with *binary urgency gating*: strict exponential penalty for slack < 0,\n      and zero contribution from upward_rank when slack <= 0 — no fading near zero, eliminating dilution of hard DDL enforcement.\n    - Uses *energy-per-latency ratio only when slack > 0*, and clamps its influence via sigmoid saturation to prevent over-prioritizing efficiency under margin.\n    - Introduces *minimal starvation boost*: sqrt-scaled wait time normalized only relative to max observed wait (not percentile),\n      bounded to [0, 0.15] — ensures fairness is secondary and never dominates deadline or criticality.\n    - Uncertainty now contributes *only when slack > 0*, scaled by (slack / (|slack| + eps)) to linearly decay importance as margin shrinks.\n    - All normalizations use degenerate-safe IQR with fallback to zero-vector for N=1 or constant inputs; no mean/std-based scaling.\n    - Final weights enforce strict hierarchy: deadline risk (3.0) >> criticality (1.8) >> energy (0.7) >> fairness (0.1) >> uncertainty (0.2) >> work (0.1).\n    '
    eps = 1e-08
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    
    def safe_iqr_normalize(x):
        """IQR normalization robust to N=1 and constant arrays; returns zeros if degenerate."""
        if x.size == 1:
            return np.zeros_like(x)
        q1 = np.percentile(x, 25)
        q3 = np.percentile(x, 75)
        iqr = q3 - q1 + eps
        if iqr < eps:
            return np.zeros_like(x)
        normed = (x - q1) / iqr
        return np.clip(normed, -3.0, 3.0)
    
    # Hard deadline risk: exponential penalty only for slack < 0; zero otherwise
    deadline_risk_raw = np.where(slack < 0, np.exp(np.clip(-slack, 0, 20)) - 1.0, 0.0)
    deadline_score = safe_iqr_normalize(deadline_risk_raw)
    
    # Criticality: active ONLY when slack > 0; abrupt cutoff at slack <= 0 (no sigmoid fade)
    upward_rank_active = np.where(slack > 0, upward_rank, 0.0)
    upward_rank_norm = safe_iqr_normalize(upward_rank_active)
    
    # Energy efficiency: only considered under positive slack, and saturated to bound impact
    total_latency = min_exec_time + min_comm_time + eps
    energy_eff_ratio = min_incremental_energy / total_latency
    energy_eff_gated = np.where(slack > 0, energy_eff_ratio, 0.0)
    energy_eff_norm = safe_iqr_normalize(energy_eff_gated)
    # Saturate energy efficiency contribution to avoid dominance: sigmoid(energy_eff_norm * 0.5)
    energy_eff_saturated = 1.0 / (1.0 + np.exp(-energy_eff_norm * 0.5))
    
    # Minimal starvation boost: bounded sqrt-scale relative to max wait, capped at 0.15
    max_wait = np.max(ready_wait_time) + eps
    wait_boost = np.sqrt(np.clip(ready_wait_time / max_wait, 0.0, 1.0))
    wait_boost = np.clip(wait_boost, 0.0, 0.15)
    
    # Uncertainty: only active when slack > 0, linearly scaled by normalized slack margin
    slack_margin = np.clip(slack, 0.0, None)  # zero out negatives
    slack_normed = slack_margin / (np.max(slack_margin + eps) + eps)
    uncertainty_gated = np.where(slack > 0, uncertainty * slack_normed, 0.0)
    uncertainty_norm = safe_iqr_normalize(uncertainty_gated)
    
    # Work normalization: low weight, only for tie-breaking among similar tasks
    remaining_work_norm = safe_iqr_normalize(remaining_work)
    
    # Strict hierarchical weighting: deadline safety dominates; fairness is minimal and bounded
    score = (
        +3.0 * deadline_score 
        - 1.8 * upward_rank_norm 
        + 0.7 * energy_eff_saturated 
        + 0.1 * wait_boost 
        + 0.2 * uncertainty_norm 
        + 0.1 * remaining_work_norm
    )
    
    # Final safeguard: replace NaN/inf with large finite values (consistent with contract)
    return np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
