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

    '''
    v3 priority rule: Deadline-feasibility-first hierarchy with:
      - Urgency dominance enforced via hard zero-slack gating (not soft threshold)
      - Criticality×SlackSens decoupled from energy layer to preserve feasibility focus
      - Energy penalty now *inverted* under safe slack (>0) to promote low-energy tasks
        only when deadline margin exists — eliminates risk-ignoring energy minimization
      - Work-density bonus activated *only when slack > 0*, not relative to median
      - Wait pressure replaced by starvation-aware *relative age* normalized by critical path
      - Unified robust normalization with strict finite-domain enforcement and N=1 safety
      - All components bounded, clipped, and nan-cleaned before combination
    '''
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
            x_finite = x_clean[np.isfinite(x_clean)]
            if x_finite.size == 0:
                return np.zeros_like(x_clean)
            p05 = np.percentile(x_finite, 5.0, method='midpoint')
            p95 = np.percentile(x_finite, 95.0, method='midpoint')
            x_clipped = np.clip(x_clean, p05, p95)
            x_min = np.min(x_clipped)
            x_max = np.max(x_clipped)
            if x_max - x_min < eps:
                return np.zeros_like(x_clean)
            return (x_clipped - x_min) / (x_max - x_min + eps)
        z = (x_clean - med) / (mad + eps)
        return np.clip(z, -3.0, 3.0)

    duration = min_exec_time + min_comm_time + eps
    rel_slack = np.divide(slack, duration, out=np.zeros_like(slack), where=duration != 0)
    rel_slack = np.nan_to_num(rel_slack, nan=0.0, posinf=0.0, neginf=0.0)

    # Hard urgency gating: zero or negative slack → highest priority (lowest score)
    is_urgent = (slack <= eps).astype(np.float64)

    # Critical latency: duration × upward_rank, normalized for comparability
    critical_latency_raw = duration * upward_rank
    norm_critical_latency = robust_mad_norm(critical_latency_raw)

    # Slack sensitivity: peaked near slack=0, bounded in [0,1], smooth and symmetric
    slack_norm = np.divide(slack, duration + eps, out=np.zeros_like(slack), where=duration != 0)
    slack_norm = np.nan_to_num(slack_norm, nan=0.0)
    # tanh-based bump: peaks at slack=0, falls to ~0.1 at ±2σ
    slack_sensitivity = 0.5 * (np.tanh(-slack_norm / 0.3) + 1.0)
    critical_slack_sensitivity = upward_rank * slack_sensitivity

    # Uncertainty-regularized effective duration: sigmoid-weighted power scaling
    uncertainty_alpha = 0.65
    scaled_uncertainty = np.power(np.maximum(uncertainty, eps), uncertainty_alpha)
    unc_sig = 1.0 / (1.0 + np.exp(-uncertainty))
    effective_duration = duration * (1.0 + scaled_uncertainty * unc_sig + eps)

    # Risk-adjusted energy density: lower is better *only when slack > 0*
    risk_energy_density = np.divide(min_incremental_energy, effective_duration,
                                    out=np.zeros_like(min_incremental_energy),
                                    where=effective_duration != 0)
    risk_energy_density = np.nan_to_num(risk_energy_density, nan=0.0, posinf=0.0, neginf=0.0)
    norm_energy_density = robust_mad_norm(risk_energy_density)

    # Energy component: inverted penalty under safe slack (>0), zero otherwise
    slack_safe_mask = (slack > eps).astype(np.float64)
    energy_score = slack_safe_mask * (-0.32 * norm_energy_density)  # promote low-energy when safe

    # Work-density fairness: activate only under positive slack (not relative median)
    work_safe_mask = (slack > eps).astype(np.float64)
    norm_remaining_work = robust_mad_norm(remaining_work)
    work_density_bonus = -0.11 * norm_remaining_work * work_safe_mask

    # Starvation-aware wait pressure: relative age w.r.t. critical path length
    # Avoids bias toward long-waiting tasks when deadlines are tight
    cp_length_estimate = duration * upward_rank + eps
    relative_age = np.divide(ready_wait_time, cp_length_estimate,
                             out=np.zeros_like(ready_wait_time), where=cp_length_estimate != 0)
    relative_age = np.nan_to_num(relative_age, nan=0.0, posinf=0.0, neginf=0.0)
    relative_age = np.clip(relative_age, 0.0, 10.0)
    norm_relative_age = robust_mad_norm(relative_age)
    # Only apply wait pressure when slack is non-urgent (i.e., slack > 0)
    wait_pressure = (1.0 - is_urgent) * 0.13 * norm_relative_age

    # Priority layering: Urgency dominates; then criticality×slacksens; then safe energy; then work; then wait
    score = np.full(N, 0.0, dtype=np.float64)
    score = np.where(is_urgent, -1000000000000.0, score)
    score = np.where(is_urgent, score,
                     score + 0.47 * norm_critical_latency * critical_slack_sensitivity)
    score = np.where(is_urgent, score, score + energy_score)
    score = np.where(is_urgent, score, score + work_density_bonus)
    score = np.where(is_urgent, score, score + wait_pressure)

    # Final bounds & cleanup
    score = np.clip(score, -1000000000000.0, 1000000000000.0)
    score = np.nan_to_num(score, nan=1000000000000.0, posinf=1000000000000.0, neginf=-1000000000000.0)

    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
