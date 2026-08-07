import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule: replaces sigmoid urgency with robust slack-based linear penalty,
    removes inactive parameters (slack_sigmoid_* and wait_saturation_time), simplifies critical bonus,
    introduces normalized bottleneck coupling via (exec+comm) * upward_rank only (no remaining_work),
    and adds anti-starvation wait-term using clipped log scaling.
    
    Key structural changes:
      - Replaces fragile sigmoid urgency with bounded linear slack penalty: max(0, -slack) + eps
      - Removes all inactive parameters (slack_sigmoid_offset/steepness, sigmoid_clip_bound, 
        critical_rank_percentile, wait_saturation_time) per counterfactual evidence
      - Simplified bottleneck: uses (exec+comm) * upward_rank only — avoids over-coupling with remaining_work
      - Anti-starvation: log-scaled ready_wait_time with hard clip to prevent unbounded growth
      - DDL-protection gate now multiplies urgency penalty (not uncertainty) when both conditions hold
      - All features normalized via IQR with elite-tuned percentiles
      - Final score prioritizes deadline safety > bottleneck release > energy > fairness
    """
    eps = 0.002483741060231933
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    def iqr_normalize(x):
        x = np.copy(x)
        q_low = np.percentile(x, 37.70788713995337)
        q_high = np.percentile(x, 86.64917013410971)
        iqr = q_high - q_low
        center = np.median(x)
        denom = iqr if iqr > eps else np.max(np.abs(x - center)) + eps
        return (x - center) / (denom + eps)
    urgency_penalty = np.maximum(-slack, 0.0) + eps
    norm_urgency = iqr_normalize(urgency_penalty)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / duration
    norm_energy_eff = iqr_normalize(energy_per_duration)
    bottleneck_pressure = duration * (upward_rank + eps)
    norm_bottleneck = iqr_normalize(bottleneck_pressure)
    norm_rank = iqr_normalize(upward_rank)
    critical_bonus = norm_rank
    wait_clipped = np.clip(ready_wait_time, 0.0, 84.89719143276963)
    wait_log = np.log1p(wait_clipped)
    norm_wait = iqr_normalize(wait_log)
    median_slack = np.median(slack)
    max_uncertainty = np.max(uncertainty) if N > 0 else 1.0
    ddl_protection_active = ((slack < median_slack) & (uncertainty > 0.5 * max_uncertainty)).astype(float)
    score = norm_urgency + 0.6496048044617384 * norm_bottleneck + 1.4540528973205389 * norm_energy_eff - critical_bonus - norm_wait + ddl_protection_active * norm_urgency
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
