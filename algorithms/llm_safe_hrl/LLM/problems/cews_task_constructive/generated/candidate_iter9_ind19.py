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

    eps = 1e-08
    # Safe casting and NaN/inf handling: use bounded finite defaults, preserve sign semantics for slack
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=0.0, posinf=1e9, neginf=eps)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=0.0, posinf=1e9, neginf=eps)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=0.0, posinf=1e9, neginf=eps)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=1e9, neginf=-1e9)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=0.0, posinf=1e9, neginf=eps)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=0.0, posinf=1e9, neginf=eps)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=1e9, neginf=eps)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=0.0, posinf=1e9, neginf=eps)
    
    N = len(slack)
    
    # Robust rank-based standardization: handles N=1, flat arrays, outliers; avoids mean/std collapse
    def rank_standardize(x):
        if N == 1:
            return np.array([0.0])
        # Stable ranks: argsort(argsort) + 1 ensures unique ordering even with ties (stable sort)
        ranks = np.argsort(np.argsort(x, kind='stable')) + 1.0
        mean_rank = np.mean(ranks)
        std_rank = np.std(ranks, ddof=0) if N > 1 else 1.0
        # Bounded denominator to prevent division by near-zero std in flat arrays
        return (ranks - mean_rank) / (std_rank + eps)
    
    # === Urgency: Hard-deadline-first enforcement ===
    # Asymmetric sigmoid: steeper penalty for violation (slack < 0), gentler reward for slack
    # tau adapts to observed slack scale but floors at minimum sensitivity
    median_slack_abs = np.abs(np.median(slack)) if N > 1 else 1.0
    tau = np.maximum(median_slack_abs, 0.1)
    # Use piecewise-linear urgency for numerical stability & interpretability near zero
    urgency_linear = np.where(slack >= 0, 
                              np.clip(slack / (tau + eps), 0.0, 1.0),
                              -np.clip(slack / (tau + eps), -5.0, 0.0))
    # Combine smooth sigmoid for tail behavior + linear core for robustness
    urgency_sigmoid = 1.0 / (1.0 + np.exp(-2.0 * slack / (tau + eps)))
    urgency = np.where(np.abs(slack) < 0.01, urgency_linear, urgency_sigmoid)
    
    # Hard violation penalty: linear penalty proportional to lateness depth, capped per-task
    abs_slack_violation = np.maximum(-slack, 0.0)
    hard_penalty = np.clip(abs_slack_violation * 8.0, 0.0, 100.0)
    deadline_urgency = urgency + hard_penalty
    
    norm_urgency = rank_standardize(deadline_urgency)
    urgency_term = -3.5 * norm_urgency  # Stronger weight to enforce DDL hardness
    
    # === Criticality-aware energy efficiency gating ===
    # Compute critical density: importance per unit execution effort
    exec_effort = np.maximum(min_exec_time, eps)
    critical_density = upward_rank * (remaining_work / exec_effort)
    
    # Latency cost = compute + comm; avoid division by zero
    latency_cost = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_latency = min_incremental_energy / latency_cost
    
    # Efficiency score: energy per critical unit — only penalized when violating
    eff_per_crit = energy_per_latency / (critical_density + eps)
    # Gate: only apply high penalty when slack < 0; otherwise neutral baseline
    eff_baseline = np.mean(eff_per_crit + eps)
    gated_eff_score = np.where(slack < 0, eff_per_crit, eff_baseline)
    
    norm_eff = rank_standardize(gated_eff_score)
    efficiency_term = -1.0 * norm_eff  # Increased weight: prioritize energy only after DDL safety
    
    # === Aging-aware boost: prevents starvation without biasing against short tasks ===
    # Relative wait quantile using stable searchsorted with tie-breaking offset
    if N == 1:
        rel_wait_quantile = np.array([0.5])
    else:
        sorted_wait = np.sort(ready_wait_time, kind='stable')
        # Add 0.5 for midpoint interpolation in quantile estimation
        rel_wait_quantile = (np.searchsorted(sorted_wait, ready_wait_time, side='right') + 0.5) / N
    # Boost only tasks with rising urgency AND waiting long — avoids boosting idle low-urgency tasks
    aging_boost = urgency * rel_wait_quantile
    norm_aging = rank_standardize(aging_boost)
    aging_term = -0.5 * norm_aging  # Slightly increased weight for fairness
    
    # === Uncertainty-aware risk amplification ===
    # Amplify uncertainty penalty only when slack is tight or violated
    # Use normalized slack distance to deadline: [0,1] for safe, >1 for violated
    slack_distance = np.maximum(-slack, 0.0) / (tau + eps)
    risk_factor = 1.0 + np.clip(slack_distance, 0.0, 5.0)
    risk_amplified_uncertainty = uncertainty * risk_factor
    uncertainty_penalty = risk_amplified_uncertainty * abs_slack_violation
    norm_uncertainty = rank_standardize(uncertainty_penalty)
    uncertainty_term = 0.6 * norm_uncertainty  # Increased weight for risk-awareness
    
    # === Slack-conditioned critical leverage: prioritize critical work *only* when slack permits ===
    # Leverage = criticality × slack margin → rewards scheduling high-impact tasks early *when possible*
    slack_margin = np.clip(slack / (tau + eps), 0.0, 2.0)  # Bounded positive margin
    critical_leverage = upward_rank * slack_margin
    norm_critical_leverage = rank_standardize(critical_leverage)
    critical_leverage_term = -0.7 * norm_critical_leverage  # Slightly stronger than v1
    
    # === Final score: sum all terms; ensure finite output ===
    score = urgency_term + efficiency_term + aging_term + uncertainty_term + critical_leverage_term
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    
    # Guarantee shape (N,) — critical for single-task edge case
    assert score.shape == (N,), f"Expected shape {(N,)}, got {score.shape}"
    return score
