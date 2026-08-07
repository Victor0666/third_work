import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with tanh-based urgency, additive uncertainty coupling, and clamped DDL-risk sensitivity.
    
    Key structural improvements:
      - Replaces sigmoid urgency with numerically stable tanh: `tanh(scale * slack)` → [-1,1], then clamped to avoid pathological penalties.
      - Converts bottleneck uncertainty coupling from multiplicative to *additive*: `+ coupling * uncertainty`, preventing explosion under high uncertainty.
      - Introduces explicit `urgency_clamp_lower` to ensure tasks with marginally negative slack aren't over-penalized — preserves fairness while maintaining hard-DDL safety.
      - Retains dynamic feasibility gating and exponentiated critical-path under joint DDL-risk.
      - Keeps anti-starvation wait boost activated only when slack < 0.
      - All operations remain finite, deterministic, and numerically safe.
    """
    eps = 6.251637949560327e-05
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
        q_low = np.percentile(x, 11.422587421199518)
        q_high = np.percentile(x, 69.84560504831411)
        iqr = q_high - q_low
        center = np.median(x)
        if iqr < eps:
            x_min, x_max = (np.min(x), np.max(x))
            denom = x_max - x_min
            if denom < eps:
                return np.zeros_like(x)
            return (x - x_min) / (denom + eps)
        else:
            denom = iqr
            return (x - center) / (denom + eps)
    tanh_input = 3.35716723877931 * slack
    urgency = np.tanh(tanh_input)
    urgency = np.clip(urgency, -0.8656606683083671, 1.0)
    norm_urgency = adaptive_normalize(urgency)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / duration
    norm_energy_eff = adaptive_normalize(energy_per_duration)
    bottleneck_pressure = duration * upward_rank * (1.0 + urgency + eps) + 0.1610126410316432 * uncertainty * duration
    norm_bottleneck = adaptive_normalize(bottleneck_pressure)
    if N == 1:
        median_wait = ready_wait_time[0]
    else:
        median_wait = np.median(ready_wait_time)
    wait_ramp = np.clip(ready_wait_time / (median_wait + eps), 0.0, 1.0)
    norm_wait = adaptive_normalize(wait_ramp)
    slack_penalty_mask = (slack < 0.0).astype(float)
    boosted_wait = norm_wait * (1.0 + slack_penalty_mask * 0.7854091520813999)
    slack_iqr = np.percentile(slack, 69.84560504831411) - np.percentile(slack, 11.422587421199518)
    dynamic_feasibility_threshold = np.median(slack) + 0.899377688403762 * slack_iqr
    feasibility_gate = np.where(slack > dynamic_feasibility_threshold, 0.0, 1.0)
    median_slack = np.median(slack)
    max_uncertainty = np.max(uncertainty) if N > 0 else eps
    ddl_risk_gate = ((slack < median_slack) & (uncertainty > 0.6790691366625087 * max_uncertainty)).astype(float)
    critical_path_strength = (upward_rank * remaining_work + eps) ** 1.5347777527776238
    norm_critical_path = adaptive_normalize(critical_path_strength)
    score = norm_urgency + 1.0300949221828504 * ddl_risk_gate * norm_critical_path + feasibility_gate * 1.3267191541599366 * norm_energy_eff - boosted_wait
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
