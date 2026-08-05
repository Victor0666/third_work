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
    v4: Refined deadline-hard priority rule with absolute urgency preservation,
         decoupled risk modeling, starvation-activated fairness, and energy-efficiency gating.
    
    Key self-evolution improvements:
    - Restores ABSOLUTE slack-based urgency (not relative) to enforce hard DDL adherence:
        arctan(slack/10) for positive slack, linear ramp from 0→2 over [-10,0], exp(|slack|-10) for < -10
    - Decouples risk penalty: only uncertainty * |slack| when slack < 0 — removes spurious multiplicative amplification
    - Fairness now STARVATION-ACTIVATED: tanh(wait_time / max(1, -min(slack, 0)+eps)) → boosts fairness most when slack is most negative
    - Energy-efficiency (CED) gated by slack >= 0 AND upward_rank > 0 → prevents energy optimization on violating paths
    - All terms normalized via safe_mad_normalize with strict outlier clipping and N=1 handling
    - Final weights reinforce hierarchy: deadline (5.3) >> energy (2.9) >> critical-path (1.3) >> risk (0.5) >> fairness (0.08)
    - Explicit finite-range clamping and nan_to_num to guarantee determinism and finiteness
    """
    eps = 1e-08
    def clean_array(x):
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    
    min_exec_time = clean_array(np.asarray(min_exec_time, dtype=float))
    min_comm_time = clean_array(np.asarray(min_comm_time, dtype=float))
    min_incremental_energy = clean_array(np.asarray(min_incremental_energy, dtype=float))
    slack = clean_array(np.asarray(slack, dtype=float))
    upward_rank = clean_array(np.asarray(upward_rank, dtype=float))
    remaining_work = clean_array(np.asarray(remaining_work, dtype=float))
    ready_wait_time = clean_array(np.asarray(ready_wait_time, dtype=float))
    uncertainty = clean_array(np.asarray(uncertainty, dtype=float))
    
    def safe_mad_normalize(x):
        """Robust MAD normalization: handles N=1, constants, outliers; clips to [-4.0, 4.0]"""
        if x.size == 1:
            return np.zeros_like(x)
        x_clipped = np.clip(x, -1000000.0, 1000000.0)
        med = np.median(x_clipped)
        mad = np.median(np.abs(x_clipped - med)) + eps
        if mad < eps:
            return np.zeros_like(x)
        normed = (x_clipped - med) / mad
        return np.clip(normed, -4.0, 4.0)
    
    # Absolute urgency: preserves hard-DDL signal strength
    arctan_urgency = 0.5 + 1.0 / np.pi * np.arctan(np.where(slack >= 0, slack / 10.0, 0.0))
    linear_violation = np.where((slack < 0) & (slack >= -10), 1.0 + (-slack) / 10.0, 0.0)
    severe_violation = np.where(slack < -10, np.exp(np.clip(-slack - 10, 0.0, 20.0)), 0.0)
    deadline_risk_raw = arctan_urgency + linear_violation + severe_violation
    deadline_score = safe_mad_normalize(deadline_risk_raw)
    
    # CED: only active on feasible (non-violating) and critical paths
    ced_gate = ((slack >= 0) & (upward_rank > eps)).astype(float)
    ced_numerator = upward_rank * remaining_work + eps
    ced_denominator = (min_incremental_energy + eps) * (min_exec_time + min_comm_time + eps)
    ced_raw = np.where(ced_gate > 0.0, ced_numerator / ced_denominator, 0.0)
    ced_norm = safe_mad_normalize(ced_raw)
    
    # Upward rank: only active when slack > 0 (to avoid misleading importance on violating paths)
    upward_rank_active = np.where((slack > 0) & (remaining_work > eps), upward_rank, 0.0)
    upward_rank_norm = safe_mad_normalize(upward_rank_active)
    
    # Risk penalty: purely slack-driven uncertainty cost — no coupling with urgency
    risk_penalty_raw = np.where(slack < 0, (-slack) * np.clip(uncertainty, 0.0, 0.7), 0.0)
    risk_penalty_norm = safe_mad_normalize(risk_penalty_raw)
    
    # Starvation-activated fairness: strongest boost when slack is most negative (max starvation risk)
    min_slack_neg = np.clip(-np.min(slack), 0.0, 1000000.0) + eps
    wait_scale = np.clip(ready_wait_time / min_slack_neg, 0.0, 20.0)
    fairness_boost = np.tanh(wait_scale)  # saturates smoothly, avoids explosion
    fairness_boost = np.clip(fairness_boost, 0.0, 0.15)  # bounded boost
    fairness_norm = safe_mad_normalize(fairness_boost)
    
    # Weighted linear combination — strict objective hierarchy
    w_deadline = 5.3
    w_ced = 2.9
    w_upward = 1.3
    w_risk = 0.5
    w_fairness = 0.08
    
    score = (
        w_deadline * deadline_score
        - w_ced * ced_norm
        - w_upward * upward_rank_norm
        + w_risk * risk_penalty_norm
        + w_fairness * fairness_norm
    )
    
    # Final safeguard: ensure finite, deterministic output
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    score = np.clip(score, -1e12, 1e12)
    
    return score
