import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining strengths from both parents:
    
    Key structural improvements:
      - Retains Parent 2's unified bottleneck (duration+comm)*upward_rank*(1+urgency)*(1+uncertainty)*(energy+eps)
        but adds explicit energy-uncertainty coupling via tunable multiplier, decoupling it from bottleneck scaling.
      - Introduces critical-path bonus (from Parent 1) activated only above percentile threshold on normalized upward_rank,
        applied additively to urgency signal — preserves deadline primacy while boosting top-critical tasks.
      - Replaces Parent 2's slack-median-based DDL gate with robust joint condition: (slack < median_slack) AND
        (uncertainty > threshold * max_uncertainty), avoiding sensitivity to zero/median artifacts.
      - Uses clipped tanh urgency [0,1] (Parent 2) but computes scale as median(|slack| + eps) for stability.
      - Anti-starvation uses normalized wait time boosted under DDL risk, same as Parent 2.
      - All normalizations use adaptive IQR with tunable percentiles; no fallbacks or clipping beyond bounds.
      - Critical bonus subtracted (since lower score = higher priority) to directly elevate critical tasks.
    """
    eps = 0.0006857670771070205
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    def adaptive_normalize(x):
        x = np.copy(x)
        q_low = np.percentile(x, 39.56821103767055)
        q_high = np.percentile(x, 72.38967627234202)
        iqr = q_high - q_low + eps
        center = np.median(x)
        return (x - center) / (iqr + eps)
    abs_slack = np.abs(slack)
    scale = np.median(abs_slack) + eps
    tanh_urgency = np.tanh(-slack / scale)
    urgency = np.clip((tanh_urgency + 1.0) / 2.0, 0.0, 1.0)
    norm_urgency = adaptive_normalize(urgency)
    norm_rank = adaptive_normalize(upward_rank)
    if N == 1:
        rank_percentile = np.array([1.0])
    else:
        sorted_norm_ranks = np.sort(norm_rank)
        rank_idx = np.searchsorted(sorted_norm_ranks, norm_rank, side='right')
        rank_percentile = rank_idx / (N + eps)
    critical_gate = np.where(rank_percentile >= 0.7787320299225031, 1.0, 0.0)
    critical_bonus = critical_gate * norm_rank
    duration = min_exec_time + min_comm_time
    coupled_energy = min_incremental_energy * (1.0 + 0.933093732318587 * uncertainty + eps)
    bottleneck_pressure = duration * upward_rank * (1.0 + urgency) * (1.0 + uncertainty) * (coupled_energy + eps)
    norm_bottleneck = adaptive_normalize(bottleneck_pressure)
    norm_wait = adaptive_normalize(ready_wait_time)
    slack_penalty_mask = (slack < 0.0).astype(float)
    boosted_wait = norm_wait * (1.0 + slack_penalty_mask * 0.04069687822334988)
    max_uncertainty = np.max(uncertainty) if N > 0 else eps
    median_slack = np.median(slack)
    ddl_risk_gate = ((slack < median_slack) & (uncertainty > 0.6317307251383608 * max_uncertainty)).astype(float)
    ddl_risk_amplification = ddl_risk_gate * norm_rank
    score = norm_urgency - critical_bonus + 0.12432104542176146 * norm_bottleneck + ddl_risk_amplification - boosted_wait
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
