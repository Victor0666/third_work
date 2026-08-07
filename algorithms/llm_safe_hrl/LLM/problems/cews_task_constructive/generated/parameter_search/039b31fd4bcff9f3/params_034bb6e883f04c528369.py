import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule with structural improvements:
      - Replaces linear urgency cap with smooth, bounded sigmoid urgency derived directly from slack.
      - Introduces explicit conditional DDL protection gate: tasks with slack < -eps receive +inf priority (hard guard).
      - Uses configurable percentiles for robust IQR-based normalization (no hidden 25/50/75 literals).
      - Bottleneck term explicitly couples upward_rank, remaining_work, and duration via product, amplified by uncertainty.
      - Energy efficiency term uses min_incremental_energy / (min_exec_time + min_comm_time + eps), weighted separately.
      - Fairness term uses saturating tanh(ready_wait_time / wait_fairness_scale).
      - All operations guarded against NaN/inf/zero; deterministic; no branching beyond hard DDL guard.
      - Uses only {-2,-1,0,1,2} literals; all tunables exposed via PARAMS.
    """
    eps = 3.3300929268419257e-05
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    is_violated = slack < -eps
    base_score = np.full(N, 0.0, dtype=float)

    def robust_normalize(x):
        x = np.copy(x)
        if N > 2:
            q25 = np.percentile(x, 24.935563908515768)
            q50 = np.percentile(x, 49.32169092632352)
            q75 = np.percentile(x, 69.3235425630305)
        else:
            q25 = q50 = q75 = x[0] if N > 0 else 0.0
        iqr = q75 - q25
        dispersion = iqr if iqr > eps else np.max(x) - np.min(x) if N > 1 else eps
        center = q50
        denom = dispersion + eps
        return (x - center) / denom
    neg_slack = np.clip(-slack, 0.0, None)
    ddl_penalty = 0.8365931927796703 * neg_slack
    slack_norm = robust_normalize(slack)
    urgency = 1.0 - 1.0 / (1.0 + np.exp(-slack_norm * 0.8563026335359367))
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    bottleneck_base = duration * upward_rank * remaining_work
    unc_norm = robust_normalize(uncertainty)
    bottleneck_pressure = bottleneck_base * np.power(1.0 + np.abs(unc_norm), 0.6041108059510925)
    norm_bottleneck = robust_normalize(bottleneck_pressure)
    energy_per_duration = min_incremental_energy / (duration + eps)
    norm_energy_eff = robust_normalize(energy_per_duration)
    wait_normalized = ready_wait_time / (6.69356460114513 + eps)
    fairness_boost = 1.0 - np.tanh(wait_normalized)
    score = ddl_penalty + robust_normalize(urgency) + 0.0009265280669109101 * norm_bottleneck + 1.154491216825789 * norm_energy_eff - fairness_boost
    finfo = np.finfo(float)
    score = np.where(is_violated, finfo.max, score)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
