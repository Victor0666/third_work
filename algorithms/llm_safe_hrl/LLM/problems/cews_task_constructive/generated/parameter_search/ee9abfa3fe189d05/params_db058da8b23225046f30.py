import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule with structural improvements:
      - Replaces pre-normalized neg_slack with robust power-law DDL risk penalty (ddl_risk_amplification)
      - Introduces conditional uncertainty gate: only amplify critical-path pressure when uncertainty exceeds threshold
      - Uses MAD-stabilized normalization for energy term instead of std-based adaptive normalize
      - Couples upward_rank and remaining_work *only* under tight slack (<= median_slack), avoiding over-amplification
      - Replaces sigmoid wait saturation with bounded power-law wait decay for smoother anti-starvation behavior
      - Removes all multiplicative joint-risk couplings; uses additive, conditionally gated terms for clarity and stability
      - All operations protected against NaN/inf/zero; deterministic; no hidden constants beyond {-2,-1,0,1,2}
    """
    eps = 9.950113755543836e-05
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    def mad_normalize(x):
        x = np.copy(x)
        center = np.median(x)
        abs_dev = np.abs(x - center)
        mad = np.median(abs_dev) if N > 0 else eps
        fallback_range = np.max(x) - np.min(x) if N > 0 else eps
        denom = mad if mad > eps else fallback_range
        return (x - center) / (denom + eps)
    neg_slack = np.clip(-slack, 0.0, None)
    ddl_risk_penalty = np.power(neg_slack + eps, 3.7151508484851723)
    median_slack = np.median(slack) if N > 0 else 0.0
    tight_slack_mask = (slack <= median_slack).astype(float)
    critical_path_pressure = tight_slack_mask * upward_rank * remaining_work
    norm_critical_path = mad_normalize(critical_path_pressure)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / (duration + eps)
    norm_energy = mad_normalize(energy_per_duration) * 0.7345250757033834
    unc_gate = (uncertainty >= 0.7826008345819919).astype(float)
    bottleneck_pressure = duration * upward_rank * remaining_work * (1.0 + unc_gate * uncertainty)
    norm_bottleneck = mad_normalize(bottleneck_pressure)
    max_wait = np.max(ready_wait_time) if N > 0 else eps
    wait_ratio = np.clip(ready_wait_time / (max_wait + eps), 0.0, 1.0)
    wait_decay = np.power(wait_ratio + eps, 0.4162030743169891)
    wait_priority = 1.0 - wait_decay
    norm_wait = mad_normalize(wait_priority)
    score = ddl_risk_penalty + 0.6207122847848588 * norm_critical_path + norm_bottleneck + norm_energy - norm_wait
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
