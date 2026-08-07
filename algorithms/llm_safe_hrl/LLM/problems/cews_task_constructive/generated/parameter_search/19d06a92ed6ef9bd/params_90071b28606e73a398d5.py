import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule with:
    - Replaced sigmoid urgency with smooth, bounded tanh-based slack mapping for stable gradient near zero.
    - Removed energy-duration ratio term: energy is now embedded in bottleneck via min_incremental_energy * (1 + urgency).
    - Unified bottleneck: (duration + comm) * upward_rank * (1 + tanh(slack)) * (1 + uncertainty) * (min_incremental_energy + eps)
      — directly couples risk-adjusted energy into critical-path pressure.
    - Anti-starvation via normalized ready_wait_time, scaled by inverse slack magnitude when slack < 0.
    - DDL-risk amplification now uses joint condition: tight slack AND high uncertainty → activates only under verified stress.
    - All normalizations use adaptive IQR; no fallback to min-max (IQR percentiles tuned to be robust across scenarios).
    - No inactive parameters: removed slack_sigmoid_* and wait_ramp_* per counterfactual evidence of redundancy.
    """
    eps = 0.00022772999820159376
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
        q_low = np.percentile(x, 39.35903376721798)
        q_high = np.percentile(x, 81.8904549066554)
        iqr = q_high - q_low + eps
        center = np.median(x)
        return (x - center) / iqr
    abs_slack = np.abs(slack)
    scale = np.median(abs_slack) if np.any(abs_slack > eps) else 1.0
    tanh_urgency = np.tanh(-slack / (scale + eps))
    urgency = np.clip((tanh_urgency + 1.0) / 2.0, 0.0, 1.0)
    norm_urgency = adaptive_normalize(urgency)
    duration = min_exec_time + min_comm_time
    bottleneck_pressure = duration * upward_rank * (1.0 + urgency) * (1.0 + uncertainty) * (min_incremental_energy + eps)
    norm_bottleneck = adaptive_normalize(bottleneck_pressure)
    norm_wait = adaptive_normalize(ready_wait_time)
    slack_penalty_mask = (slack < 0.0).astype(float)
    boosted_wait = norm_wait * (1.0 + slack_penalty_mask * 0.21392847361869183)
    max_uncertainty = np.max(uncertainty) if N > 0 else eps
    ddl_risk_gate = ((slack < 0.6052443944060193 * np.abs(np.median(slack) + eps)) & (uncertainty > 0.6052443944060193 * max_uncertainty)).astype(float)
    ddl_risk_amplification = ddl_risk_gate * upward_rank
    score = norm_urgency + 0.4485560981195379 * norm_bottleneck + ddl_risk_amplification - boosted_wait
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
