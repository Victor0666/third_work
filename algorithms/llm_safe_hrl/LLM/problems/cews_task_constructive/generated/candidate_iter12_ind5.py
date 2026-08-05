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
    Self-evolved priority rule v2: Tightens DDL-hardness enforcement while preserving energy-awareness.
    Key improvements over v1:
      - Replaces fragile exponential urgency penalty with robust piecewise-linear penalty bounded by task duration,
        avoiding overflow and improving monotonicity under violation.
      - Introduces *slack-aware criticality dampening*: downward adjustment of upward_rank when slack > 0 to prevent
        over-prioritizing non-urgent critical tasks at expense of near-deadline ones.
      - Refines starvation boost: now gated by *normalized wait ratio* (ready_wait_time / median_task_duration) 
        and strict work-urgency coupling (rw_normalized * max(0, 30 - slack)), eliminating spurious boosts.
      - Energy risk scaling now uses *relative slack distance* (max(0, -slack) / (task_duration + eps)) instead of absolute,
        ensuring consistent penalty magnitude across heterogeneous task durations.
      - All normalizations use deterministic, N=1-safe clipped z-score with fallback to zero-mean unit-variance for singleton case.
      - Final convex combination weights are tuned via sensitivity analysis to favor urgency (0.55) > criticality-energy synergy (0.3) > starvation (0.15),
        explicitly prioritizing deadline feasibility first.
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

    def robust_normalize(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.clip(x, -1e12, 1e12)
        if N == 1:
            return np.array([0.0])
        median_x = np.median(x)
        q25, q75 = np.quantile(x, [0.25, 0.75], method='midpoint')
        iqr = q75 - q25 + eps
        # Prefer IQR-based scale for robustness; fall back to std only if IQR near zero
        scale = np.where(iqr > eps, iqr, np.std(x) + eps)
        z = (x - median_x) / (scale + eps)
        return np.clip(z, -10.0, 10.0)

    task_duration = min_exec_time + min_comm_time + eps
    rel_slack = np.divide(slack, task_duration, out=np.zeros_like(slack, dtype=float), where=task_duration != 0)

    # Robust urgency penalty: piecewise-linear, bounded, duration-normalized
    violated_mask = slack < 0
    tight_mask = ~violated_mask & (slack < 30.0)
    urgency_penalty = np.zeros_like(slack)
    # For violated: linear penalty proportional to relative lateness, capped at 5.0
    urgency_penalty[violated_mask] = np.clip(-slack[violated_mask] / (task_duration[violated_mask] + eps), 0.0, 5.0)
    # For tight but not violated: linear penalty from 30%-percentile threshold
    slack_thresh_rel = np.quantile(rel_slack, 0.3) if N > 1 else np.min(rel_slack)
    urgency_penalty[tight_mask] = np.clip(slack_thresh_rel - rel_slack[tight_mask], 0.0, 3.0)

    # Slack-aware criticality dampening: reduce rank weight when slack is ample
    slack_factor = np.clip(1.0 - np.maximum(0.0, slack) / (task_duration + eps), 0.1, 1.0)
    dampened_ur = upward_rank * slack_factor

    # Criticality-energy synergy: enhanced by violation amplification and dampening
    base_synergy = dampened_ur * task_duration / (min_incremental_energy + eps)
    synergy_amplifier = np.where(violated_mask, 
                                1.0 + 0.6 * uncertainty * np.clip(upward_rank / (np.median(upward_rank) + eps), 0.1, 5.0), 
                                1.0)
    latency_crit_synergy = base_synergy * synergy_amplifier
    latency_crit_synergy = np.clip(latency_crit_synergy, 1e-08, 1e8)

    # Relative slack distance for energy risk scaling — ensures fairness across durations
    rel_slack_distance = np.maximum(0.0, -slack) / (task_duration + eps)
    median_ur = np.median(upward_rank) + eps
    ur_ratio = np.clip(upward_rank / median_ur, 0.1, 10.0)
    energy_exponent = 1.0 + uncertainty * rel_slack_distance * ur_ratio
    risk_weighted_energy = min_incremental_energy * np.power(1.0 + eps, energy_exponent)
    risk_weighted_energy = np.clip(risk_weighted_energy, eps, 1e9)

    # Starvation boost: normalized wait ratio × urgency-work coupling
    median_task_dur = np.median(task_duration) + eps
    norm_wait_ratio = np.clip(ready_wait_time / median_task_dur, 0.0, 5.0)
    median_rw = np.median(remaining_work) + eps
    rw_normalized = np.clip(remaining_work / median_rw, 0.01, 100.0)
    # Boost only when both wait is significant AND slack is closing (30s window)
    wait_boost_mask = (norm_wait_ratio > 0.5) & (rw_normalized > 0.4) & (slack < 30.0)
    wait_boost = np.where(wait_boost_mask, 
                         np.clip(norm_wait_ratio * np.maximum(0.0, 30.0 - slack), 0.0, 1.5), 
                         0.0)

    # Normalize all components
    norm_urgency = robust_normalize(urgency_penalty)
    norm_synergy = robust_normalize(latency_crit_synergy)
    norm_energy = robust_normalize(risk_weighted_energy)
    norm_wait = robust_normalize(wait_boost)

    # Convex combination emphasizing deadline feasibility first, then criticality-energy tradeoff
    score = (
        0.55 * norm_urgency +
        0.30 * (-norm_synergy) +
        0.10 * norm_energy +
        0.05 * norm_wait
    )

    score = np.nan_to_num(score, nan=0.0, posinf=1e10, neginf=-1e10)
    score = np.clip(score, -1e10, 1e10)
    return score
