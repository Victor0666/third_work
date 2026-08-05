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
    v3 priority rule: Adaptive critical-path urgency + dynamic energy-risk coupling + calibrated starvation rescue + expanded dynamic-range normalization.
    
    Key improvements over v1:
    - Restores discriminative power: replaces [0,1] minmax with robust z-score-like normalization (centered, clipped ±8σ) to preserve fine-grained priority resolution among tight-deadline tasks.
    - Rebalances feasibility-efficiency tradeoff: reduces urgency weight to 0.48, increases synergy weight to 0.32 — prioritizing critical-path acceleration *within* feasibility bounds.
    - Tightens energy-urgency coupling: removes attenuation; instead applies *inverse slack-gated energy scaling*: energy penalty amplified only when slack > 0 (safe region), ensuring late tasks aren’t penalized for energy while preserving fairness.
    - Enhances starvation rescue: adds *criticality-aware wait ratio* (ready_wait_time / (task_duration + eps)) and gates by upward_rank percentile, preventing low-criticality starvation bias.
    - Introduces *uncertainty-as-urgency multiplier*: dur_uncertainty directly scales urgency_penalty for high-risk tasks, improving robustness under bandwidth/compute volatility.
    - Adds *deadline proximity sensitivity*: slack quantile-based urgency thresholding (q10/q50/q90) enables adaptive response across heterogeneous workflows.
    - Final weights: DDL-feasibility (0.48), synergy (0.32), comm (0.07), energy-risk (0.06), starvation (0.05), uncertainty (0.02)
    """
    eps = 1e-08
    min_exec_time = np.asarray(min_exec_time, dtype=float).copy()
    min_comm_time = np.asarray(min_comm_time, dtype=float).copy()
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float).copy()
    slack = np.asarray(slack, dtype=float).copy()
    upward_rank = np.asarray(upward_rank, dtype=float).copy()
    remaining_work = np.asarray(remaining_work, dtype=float).copy()
    ready_wait_time = np.asarray(ready_wait_time, dtype=float).copy()
    uncertainty = np.asarray(uncertainty, dtype=float).copy()
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=float)
    
    # Robust normalization: centered, scaled by MAD, clipped to ±8 — preserves discriminative range for tight tasks
    def robust_zclip(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.clip(x, -1e12, 1e12)
        if N == 1:
            return np.array([0.0])
        center = np.median(x)
        mad = np.median(np.abs(x - center)) + eps
        z = (x - center) / mad
        return np.clip(z, -8.0, 8.0)
    
    task_duration = np.maximum(min_exec_time + min_comm_time, eps)
    dur_uncertainty = np.divide(uncertainty, task_duration + eps, out=np.zeros_like(uncertainty), where=task_duration + eps != 0)
    dur_uncertainty = np.nan_to_num(dur_uncertainty, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Deadline proximity via quantiles for adaptive urgency gating
    if N > 1:
        q10, q50, q90 = np.quantile(slack, [0.1, 0.5, 0.9], method='midpoint')
    else:
        q10 = q50 = q90 = slack[0]
    
    violated_mask = slack < 0
    tight_mask = ~violated_mask & (slack < q50)
    relaxed_mask = ~violated_mask & (slack >= q50)
    
    # Base urgency: quantile-relative, capped, then scaled by criticality and uncertainty
    urgency_base = np.zeros_like(slack)
    urgency_base[violated_mask] = np.clip(-slack[violated_mask] / (task_duration[violated_mask] + eps), 0.0, 12.0)
    urgency_base[tight_mask] = np.clip((q50 - slack[tight_mask]) / (task_duration[tight_mask] + eps), 0.0, 5.0)
    urgency_base[relaxed_mask] = np.clip((q90 - slack[relaxed_mask]) / (task_duration[relaxed_mask] + eps), 0.0, 1.0)
    
    # Criticality and uncertainty amplification
    ur_norm = robust_zclip(upward_rank)
    ur_factor = 1.0 + 0.4 * np.clip(ur_norm, 0.0, 8.0)
    unc_factor = 1.0 + 0.6 * dur_uncertainty
    urgency_penalty = urgency_base * ur_factor * unc_factor
    
    # Energy penalty: only activated in safe region (slack > 0) to avoid punishing late tasks unfairly
    safe_mask = slack > 0
    rel_slack_distance = np.where(safe_mask, np.maximum(0.0, q90 - slack) / (task_duration + eps), 0.0)
    energy_exponent = 1.25
    risk_weighted_energy = min_incremental_energy * np.power(1.0 + uncertainty * rel_slack_distance + eps, energy_exponent)
    risk_weighted_energy = np.clip(risk_weighted_energy, eps, 1e10)
    
    # Synergy: critical-path work density, amplified under violation
    slack_factor = np.clip(1.0 - np.maximum(0.0, slack - q50) / (task_duration + eps), 0.05, 1.0)
    dampened_ur = upward_rank * slack_factor
    base_synergy = dampened_ur * task_duration / (min_incremental_energy + eps)
    synergy_amplifier = np.where(violated_mask, 1.0 + 0.9 * uncertainty, 1.0)
    latency_crit_synergy = base_synergy * synergy_amplifier
    latency_crit_synergy = np.clip(latency_crit_synergy, eps, 1e9)
    
    # Communication pressure: weighted by uncertainty and slack distance
    comm_overhead_ratio = min_comm_time / (task_duration + eps)
    slack_distance_penalty = np.clip(1.0 + np.maximum(0.0, -slack) / (np.median(task_duration) + eps), 1.0, 5.0)
    comm_pressure = comm_overhead_ratio * (1.0 + 0.7 * uncertainty) * slack_distance_penalty
    comm_pressure = np.where(violated_mask, comm_pressure * 3.5, np.where(tight_mask, comm_pressure * 1.8, comm_pressure * 0.4))
    
    # Starvation rescue: gated by criticality percentile and normalized wait intensity
    median_rw = np.median(remaining_work) + eps
    rw_normalized = np.clip(remaining_work / median_rw, 0.01, 100.0)
    norm_wait_ratio = np.clip(ready_wait_time / (task_duration + eps), 0.0, 20.0)
    # Gate by top-30% upward_rank to ensure high-criticality focus
    ur_percentile = np.percentile(upward_rank, 70) if N > 1 else upward_rank[0]
    is_starvable = (norm_wait_ratio > 1.0) & (rw_normalized > 0.25) & (upward_rank >= ur_percentile) & (slack < 240.0)
    starvation_boost = np.where(is_starvable, norm_wait_ratio * (1.0 + 0.35 * np.clip(ur_norm, 0.0, 8.0)), 0.0)
    starvation_boost = np.clip(starvation_boost, 0.0, 2.0)
    
    # Normalize all components with robust_zclip
    norm_urgency = robust_zclip(urgency_penalty)
    norm_synergy = robust_zclip(latency_crit_synergy)
    norm_energy = robust_zclip(risk_weighted_energy)
    norm_comm = robust_zclip(comm_pressure)
    norm_starvation = robust_zclip(starvation_boost)
    norm_uncertainty = robust_zclip(dur_uncertainty)
    
    # Final weighted score — smaller = higher priority
    score = (
        0.48 * norm_urgency +
        -0.32 * norm_synergy +
        0.06 * norm_energy +
        0.07 * norm_comm +
        0.05 * norm_starvation +
        0.02 * norm_uncertainty
    )
    
    score = np.nan_to_num(score, nan=0.0, posinf=1e12, neginf=-1e12)
    score = np.clip(score, -1e12, 1e12)
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
