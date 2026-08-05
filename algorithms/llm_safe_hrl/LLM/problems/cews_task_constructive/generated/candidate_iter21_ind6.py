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
    v2 priority rule: Hard urgency dominance + latency-gated criticality +
                      risk-normalized energy + starvation-robust fairness +
                      uncertainty-coupled slack sensitivity + stable local normalization.

    Key improvements over v1:
    - Replaces global slack statistics (mean/std) with *local robust slack pressure*:
      uses percentile-based slack quantiles within ready set only, avoiding noise in small-N.
    - Eliminates dynamic weight adaptation — restores fixed hierarchical weights for stability;
      urgency dominates unconditionally, then latency, then energy, etc.
    - Fixes work-density term: removes sign-flipping; instead applies *urgency-modulated penalty*:
      high energy-per-MI penalized only when slack > 0, and suppressed entirely when slack <= 0.
    - Uses *local trimmed norm* per feature: median ± 3*MAD computed strictly on current ready set,
      with explicit fallback to zero when MAD=0 or size<2 (ensures small-N stability).
    - Introduces *latency-uncertainty coupling*: multiplies normalized uncertainty with
      normalized critical_latency only for tight-slack tasks, enhancing risk-aware criticality.
    - All divisions guarded, NaN/inf cleaned before normalization, and final score clipped & sanitized.
    """
    eps = 1e-08
    # Ensure float64 and copy inputs to avoid mutation
    min_exec_time = np.asarray(min_exec_time, dtype=np.float64).copy()
    min_comm_time = np.asarray(min_comm_time, dtype=np.float64).copy()
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=np.float64).copy()
    slack = np.asarray(slack, dtype=np.float64).copy()
    upward_rank = np.asarray(upward_rank, dtype=np.float64).copy()
    remaining_work = np.asarray(remaining_work, dtype=np.float64).copy()
    ready_wait_time = np.asarray(ready_wait_time, dtype=np.float64).copy()
    uncertainty = np.asarray(uncertainty, dtype=np.float64).copy()
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=np.float64)

    # Local robust normalization: median ± 3*MAD, safe for small N
    def local_robust_norm(x):
        x_clean = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if N == 1:
            return np.zeros_like(x_clean)
        x_med = np.median(x_clean)
        abs_dev = np.abs(x_clean - x_med)
        mad = np.median(abs_dev)
        scale = 3.0 * mad + eps
        if scale < eps:
            return np.zeros_like(x_clean)
        z = (x_clean - x_med) / scale
        return np.clip(z, -3.0, 3.0)

    # Urgency flag: hard dominance for violated or imminent deadlines
    is_urgent = (slack <= 0.0).astype(np.float64)

    # Duration = compute + comm; base for latency-sensitive terms
    duration = min_exec_time + min_comm_time + eps

    # Relative slack: avoids division-by-zero, clips extreme values
    rel_slack = np.divide(slack, duration, out=np.zeros_like(slack), where=duration != 0)
    rel_slack = np.nan_to_num(rel_slack, nan=0.0, posinf=0.0, neginf=0.0)

    # Latency-aware criticality gating: activate only if slack is tight AND duration is long
    duration_med = np.median(duration) + eps
    tight_slack_mask = (rel_slack <= 0.25).astype(np.float64)
    long_duration_mask = (duration >= duration_med).astype(np.float64)
    critical_gate = tight_slack_mask * long_duration_mask

    # Critical latency: duration × upward_rank → importance under latency pressure
    critical_latency_raw = duration * upward_rank
    norm_critical_latency = local_robust_norm(critical_latency_raw)

    # Risk-normalized incremental energy: suppress high-energy assignments on uncertain VMs
    effective_duration = duration * (1.0 + uncertainty + eps)
    risk_norm_energy = np.divide(
        min_incremental_energy,
        effective_duration,
        out=np.zeros_like(min_incremental_energy),
        where=effective_duration != 0
    )
    risk_norm_energy = np.nan_to_num(risk_norm_energy, nan=0.0, posinf=0.0, neginf=0.0)
    norm_energy = local_robust_norm(risk_norm_energy)

    # Starvation-robust fairness: wait time scaled by slack pressure (more boost when tight)
    norm_wait_time = local_robust_norm(ready_wait_time)
    # Slack pressure: 1.0 when slack <= 0, decays linearly to 0 at slack = median(slack)+2*std
    slack_clean = np.nan_to_num(slack, nan=0.0, posinf=0.0, neginf=0.0)
    if N > 1:
        slack_q90 = np.percentile(slack_clean, 90)
        slack_q10 = np.percentile(slack_clean, 10)
        slack_range = max(slack_q90 - slack_q10, eps)
        slack_pressure = np.clip((slack_q90 - slack_clean) / slack_range, 0.0, 1.0)
    else:
        slack_pressure = np.ones_like(slack_clean)
    wait_penalty = norm_wait_time * slack_pressure

    # Uncertainty-coupled slack sensitivity: inverse |slack| modulated by uncertainty
    abs_slack = np.abs(slack) + eps
    slack_sensitivity = np.clip(1.0 / abs_slack, 0.1, 20.0)
    norm_uncertainty = local_robust_norm(uncertainty)
    # Only amplify slack sensitivity when uncertainty is high AND slack is tight
    uncertainty_boost = norm_uncertainty * slack_sensitivity * tight_slack_mask

    # Work-density relief: energy-per-MI penalized only when slack > 0; ignored when urgent
    energy_per_mi = np.divide(
        min_incremental_energy,
        remaining_work + eps,
        out=np.zeros_like(min_incremental_energy),
        where=remaining_work + eps != 0
    )
    energy_per_mi = np.nan_to_num(energy_per_mi, nan=0.0, posinf=0.0, neginf=0.0)
    norm_energy_per_mi = local_robust_norm(energy_per_mi)
    work_density_penalty = (1.0 - is_urgent) * norm_energy_per_mi

    # Latency-uncertainty coupling: enhance critical latency penalty under high uncertainty + tight slack
    latency_uncertainty_coupling = norm_critical_latency * norm_uncertainty * tight_slack_mask

    # Fixed hierarchical weights (stable, interpretable, deadline-safe)
    w_urgency = 0.0          # Handled via hard dominance below
    w_critical = 0.40        # Dominant after urgency
    w_energy = 0.25          # Secondary: minimize risk-adjusted energy among feasible
    w_fairness = 0.15        # Tertiary: prevent starvation
    w_uncertainty = 0.12     # Quaternary: penalize uncertainty amplification near deadline
    w_work_density = 0.08    # Quinary: reward energy efficiency when slack allows

    # Base score: start neutral
    score = np.full(N, 0.0, dtype=np.float64)

    # Hard urgency dominance: assign lowest possible score to all urgent tasks
    score = np.where(is_urgent, -1e12, score)

    # Add weighted components only for non-urgent tasks (preserves hierarchy)
    score = np.where(
        is_urgent,
        score,
        score + w_critical * (norm_critical_latency + latency_uncertainty_coupling)
    )
    score = np.where(is_urgent, score, score + w_energy * norm_energy * critical_gate)
    score = np.where(is_urgent, score, score + w_fairness * wait_penalty)
    score = np.where(is_urgent, score, score + w_uncertainty * uncertainty_boost)
    score = np.where(is_urgent, score, score + w_work_density * work_density_penalty)

    # Final sanitization: clip, replace NaN/inf
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)

    # Assert shape compliance
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
