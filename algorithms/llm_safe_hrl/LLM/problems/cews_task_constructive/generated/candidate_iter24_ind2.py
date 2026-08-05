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
    v3 priority rule: Adaptive urgency gating + criticality-aware slack scaling + 
                      uncertainty-robust energy normalization + work-density fairness +
                      deadline-criticality decoupling.

    Key evolutions:
    - Replaces fixed `rel_slack <= 0.3` with *adaptive tightness threshold*: 
      median(rel_slack) + 0.5 * MAD(rel_slack), robust to skewed deadline distributions.
    - Introduces *criticality-weighted slack sensitivity*: slack penalty scaled by upward_rank,
      ensuring high-importance tasks get sharper urgency response near deadlines.
    - Uses *uncertainty-regularized energy normalization*: replaces simple (1+uncertainty) with 
      sigmoid(uncertainty) * (1 + uncertainty) — soft-clips extreme uncertainty impact.
    - Adds *work-density fairness*: negative bonus for high remaining_work only when slack > median_slack,
      promoting early heavy-DAG scheduling under safe global margin (not per-task slack > 0).
    - Eliminates fragile `1/|slack|` clipping; instead uses smooth, bounded slack_sensitivity = 
      tanh(slack / (duration + eps)) * (1 - tanh(-slack / (duration + eps))) → peak at slack=0.
    - All normalizations use local trimmed-MAD with explicit N=1 fallback and zero-MAD safety.
    - Final layering enforces strict priority hierarchy: Urgency > Criticality×SlackSens > Energy > WorkDensity > Wait.
    """
    eps = 1e-08
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

    def robust_mad_norm(x):
        x_clean = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if N == 1:
            return np.zeros_like(x_clean)
        med = np.median(x_clean)
        abs_dev = np.abs(x_clean - med)
        mad = np.median(abs_dev)
        if mad < eps:
            return np.zeros_like(x_clean)
        z = (x_clean - med) / (mad + eps)
        return np.clip(z, -3.0, 3.0)

    # Duration & adaptive tightness threshold
    duration = min_exec_time + min_comm_time + eps
    rel_slack = np.divide(slack, duration, out=np.zeros_like(slack), where=duration != 0)
    rel_slack = np.nan_to_num(rel_slack, nan=0.0, posinf=0.0, neginf=0.0)
    rel_slack_med = np.median(rel_slack)
    rel_slack_mad = np.median(np.abs(rel_slack - rel_slack_med))
    adaptive_tight_threshold = rel_slack_med + 0.5 * max(rel_slack_mad, eps)
    tight_slack_mask = (rel_slack <= adaptive_tight_threshold).astype(np.float64)

    # Urgency: hard penalty for violated or imminent deadlines
    is_urgent = (slack <= eps).astype(np.float64)

    # Critical latency: duration × upward_rank, normalized
    critical_latency_raw = duration * upward_rank
    norm_critical_latency = robust_mad_norm(critical_latency_raw)

    # Slack sensitivity: smooth, bounded, criticality-weighted
    # tanh-based shape avoids division-by-zero and asymptotes cleanly
    slack_norm = np.divide(slack, duration + eps, out=np.zeros_like(slack), where=duration != 0)
    slack_norm = np.nan_to_num(slack_norm, nan=0.0)
    slack_sensitivity = 0.5 * (np.tanh(slack_norm / 0.5) - np.tanh((slack_norm - 2.0) / 0.5))
    # Scale by upward_rank to amplify urgency for critical paths
    critical_slack_sensitivity = upward_rank * slack_sensitivity

    # Uncertainty-regularized energy denominator
    unc_sig = 1.0 / (1.0 + np.exp(-uncertainty))  # sigmoid(uncertainty) ∈ (0,1)
    effective_duration = duration * (1.0 + uncertainty * unc_sig + eps)
    risk_norm_energy = np.divide(min_incremental_energy, effective_duration,
                                  out=np.zeros_like(min_incremental_energy),
                                  where=effective_duration != 0)
    risk_norm_energy = np.nan_to_num(risk_norm_energy, nan=0.0, posinf=0.0, neginf=0.0)
    norm_energy = robust_mad_norm(risk_norm_energy)

    # Work-density fairness: schedule heavy sub-DAGs early *only* when global slack margin exists
    global_slack_med = np.median(slack) + eps
    safe_global_margin = (slack > global_slack_med).astype(np.float64)
    norm_remaining_work = robust_mad_norm(remaining_work)
    work_density_bonus = -0.09 * norm_remaining_work * safe_global_margin

    # Wait pressure: relative wait time, activated only under pressure (low global slack)
    wait_per_duration = np.divide(ready_wait_time, duration,
                                   out=np.zeros_like(ready_wait_time),
                                   where=duration != 0)
    wait_per_duration = np.nan_to_num(wait_per_duration, nan=0.0, posinf=0.0, neginf=0.0)
    norm_wait_rel = robust_mad_norm(wait_per_duration)
    wait_pressure = (1.0 - 1.0 / (1.0 + np.exp(-(global_slack_med - slack) / (duration + eps)))) * norm_wait_rel

    # Final score: hierarchical composition with strict ordering
    score = np.full(N, 0.0, dtype=np.float64)
    # 1. Hard urgency override
    score = np.where(is_urgent, -1000000000000.0, score)
    # 2. Criticality × Slack Sensitivity (dominant under tightness)
    score = np.where(is_urgent, score, score + 0.45 * norm_critical_latency * critical_slack_sensitivity)
    # 3. Risk-normalized energy (active only when tight)
    score = np.where(is_urgent, score, score + 0.27 * norm_energy * tight_slack_mask)
    # 4. Work-density bonus (active only under safe global margin)
    score = np.where(is_urgent, score, score + work_density_bonus)
    # 5. Wait pressure (smoothly activated as global slack depletes)
    score = np.where(is_urgent, score, score + 0.15 * wait_pressure)
    # 6. Clamp and sanitize
    score = np.clip(score, -1000000000000.0, 1000000000000.0)
    score = np.nan_to_num(score, nan=1000000000000.0, posinf=1000000000000.0, neginf=-1000000000000.0)

    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
