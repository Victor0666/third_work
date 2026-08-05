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
    Self-evolved priority rule v2: Tightens deadline-hardness enforcement with adaptive urgency calibration,
    refines criticality-energy synergy via slack- and uncertainty-aware gating, and introduces starvation-avoidance
    with work-urgency coupling and MAD-normalized wait pressure. Key advances:
      - Replaces fixed slack thresholds with *adaptive urgency bands* using quantile-based dynamic thresholds
        (0.2/0.5/0.8 quantiles of rel_slack) for robustness across workload scales.
      - Introduces *criticality attenuation factor*: dampens upward_rank only when slack > median_slack,
        avoiding premature de-prioritization of moderately urgent high-rank tasks.
      - Refines communication pressure: now scaled by *uncertainty-weighted relative comm overhead* 
        (min_comm_time / task_duration) and gated by both violation and tightness, not just slack sign.
      - Unifies energy risk scoring: uses linear uncertainty amplification *plus* slack-distance exponentiation
        (1 + uncertainty * rel_slack_distance)^1.2 to strengthen penalty under severe violation while preserving smoothness.
      - Starvation boost now requires *joint sufficiency*: (normalized wait > 0.7) AND (remaining_work > 0.3 * median_rw)
        AND (slack < 60), and is capped at 1.2x normalized wait ratio to prevent dominance.
      - All normalizations use MAD with explicit fallback for N=1; all divisions guarded; all outputs finite and clipped.
      - Final weights tuned to emphasize urgency (0.52), synergy (0.33), comm pressure (0.08), energy (0.05), starvation (0.02)
        — prioritizing deadline feasibility first, then synergistic efficiency, then resource fairness.
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
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=float)

    def robust_normalize_mad(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.clip(x, -1e12, 1e12)
        if N == 1:
            return np.array([0.0])
        center = np.median(x)
        mad = np.median(np.abs(x - center)) + eps
        normed = (x - center) / mad
        return np.clip(normed, -10.0, 10.0)

    # Task duration with safety guard
    task_duration = np.maximum(min_exec_time + min_comm_time, eps)
    
    # Relative slack: slack / task_duration, safe division
    rel_slack = np.divide(slack, task_duration, out=np.zeros_like(slack), where=task_duration != 0)
    
    # Adaptive urgency bands via quantiles (robust to outliers)
    if N > 1:
        q20, q50, q80 = np.quantile(rel_slack, [0.2, 0.5, 0.8], method='midpoint')
    else:
        q20 = q50 = q80 = rel_slack[0]
    
    violated_mask = slack < 0
    tight_mask = ~violated_mask & (rel_slack < q50)  # tight: below median relative slack
    relaxed_mask = ~violated_mask & (rel_slack >= q50)
    
    # Piecewise-linear urgency penalty: stronger penalty for violation, graded penalty for tightness
    urgency_penalty = np.zeros_like(slack)
    urgency_penalty[violated_mask] = np.clip(-slack[violated_mask] / (task_duration[violated_mask] + eps), 0.0, 6.0)
    urgency_penalty[tight_mask] = np.clip(q50 - rel_slack[tight_mask], 0.0, 3.5)
    urgency_penalty[relaxed_mask] = np.clip(q80 - rel_slack[relaxed_mask], 0.0, 1.0)
    
    # Criticality attenuation: only dampen when slack > median_slack (not just positive)
    slack_factor = np.clip(1.0 - np.maximum(0.0, slack - q50 * np.median(task_duration)) / (task_duration + eps), 0.1, 1.0)
    dampened_ur = upward_rank * slack_factor
    
    # Synergy: criticality × duration / energy, amplified under violation
    base_synergy = dampened_ur * task_duration / (min_incremental_energy + eps)
    median_ur = np.median(upward_rank) + eps
    ur_ratio = np.clip(upward_rank / median_ur, 0.1, 10.0)
    synergy_amplifier = np.where(violated_mask, 1.0 + 0.7 * uncertainty * ur_ratio, 1.0)
    latency_crit_synergy = base_synergy * synergy_amplifier
    latency_crit_synergy = np.clip(latency_crit_synergy, 1e-08, 1e8)
    
    # Energy risk: (1 + uncertainty * rel_slack_distance)^exponent, then scaled by base energy
    rel_slack_distance = np.maximum(0.0, -slack) / (task_duration + eps)
    energy_exponent = 1.2
    risk_weighted_energy = min_incremental_energy * np.power(1.0 + uncertainty * rel_slack_distance + eps, energy_exponent)
    risk_weighted_energy = np.clip(risk_weighted_energy, eps, 1e9)
    
    # Communication pressure: relative comm overhead, uncertainty-weighted and slack-gated
    comm_overhead_ratio = min_comm_time / (task_duration + eps)
    comm_pressure = comm_overhead_ratio * (1.0 + 0.5 * uncertainty)
    comm_pressure = np.where(violated_mask, comm_pressure * 2.0,
                           np.where(tight_mask, comm_pressure * 0.6, comm_pressure * 0.2))
    
    # Starvation boost: joint sufficiency check + capped contribution
    median_task_dur = np.median(task_duration) + eps
    norm_wait_ratio = np.clip(ready_wait_time / median_task_dur, 0.0, 5.0)
    median_rw = np.median(remaining_work) + eps
    rw_normalized = np.clip(remaining_work / median_rw, 0.01, 100.0)
    wait_boost_mask = (norm_wait_ratio > 0.7) & (rw_normalized > 0.3) & (slack < 60.0)
    wait_boost = np.where(wait_boost_mask,
                         np.clip(norm_wait_ratio * np.maximum(0.0, 60.0 - slack) / 60.0, 0.0, 1.2),
                         0.0)
    
    # Normalize all components with MAD
    norm_urgency = robust_normalize_mad(urgency_penalty)
    norm_synergy = robust_normalize_mad(latency_crit_synergy)
    norm_energy = robust_normalize_mad(risk_weighted_energy)
    norm_comm = robust_normalize_mad(comm_pressure)
    norm_wait = robust_normalize_mad(wait_boost)
    
    # Final convex combination — urgency dominant, synergy secondary, others corrective
    score = (0.52 * norm_urgency 
             + (-0.33) * norm_synergy 
             + 0.08 * norm_comm 
             + 0.05 * norm_energy 
             + 0.02 * norm_wait)
    
    # Final safeguard: ensure finite, deterministic output
    score = np.nan_to_num(score, nan=0.0, posinf=1e10, neginf=-1e10)
    score = np.clip(score, -1e10, 1e10)
    return score
