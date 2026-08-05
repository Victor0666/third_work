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
    Self-evolved priority rule (v2) with three core advances:
      - Slack urgency refined via *adaptive thresholding*: uses task-duration-relative criticality (slack/task_duration < -0.3)
        instead of absolute seconds, improving portability across workflow scales.
      - Energy efficiency strengthened by *critical-path-aware normalization*: replaces raw upward_rank with
        normalized residual critical path length (upward_rank / max_upward_rank), preventing rank inflation bias
        in shallow DAGs and enabling fair comparison across heterogeneous sub-DAGs.
      - Starvation guard upgraded to *risk-conditioned fairness*: applies wait-time boost only when either
        (a) slack <= 0 OR (b) uncertainty > 0.8 AND ready_wait_time > p90 — explicitly coupling fairness with risk exposure.
      - Uncertainty now *gates both slack urgency AND energy penalty* but asymmetrically: amplifies slack penalty multiplicatively,
        while dampening energy density denominator additively (1 + sqrt(uncertainty)) to avoid over-penalizing high-risk low-energy tasks.
      - All robust ops use fused epsilon protection and finite filtering; no unbounded scaling or implicit assumptions.
    """
    eps = 1e-08
    min_exec_time = np.asarray(min_exec_time, dtype=float).copy()
    min_comm_time = np.asarray(min_comm_time, dtype=float).copy()
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float).copy()
    slack = np.asarray(slack, dtype=float).copy()
    upward_rank = np.asarray(upward_rank, dtype=float).copy()
    remaining_work = np.asarray(remaining_work, dtype=float).copy()
    ready_wait_time = np.asarray(ready_wait_time, dtype=float).copy()
    uncertainty = np.asarray(uncertainty, dtype=float).copy()
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=float)

    def robust_normalize(x):
        x_clean = x[np.isfinite(x) & (np.abs(x) < 1e12)]
        if len(x_clean) == 0:
            return np.zeros_like(x)
        center = np.mean(x_clean)
        scale = np.std(x_clean, ddof=1) + eps
        norm = (x - center) / scale
        return np.clip(norm, -10.0, 10.0)

    task_duration = np.maximum(min_exec_time + min_comm_time, eps)
    # Adaptive relative slack urgency: critical if slack < -30% of task duration
    rel_slack = slack / task_duration
    slack_urgency = np.where(
        rel_slack < -0.3,
        np.exp(np.clip(-rel_slack - 0.3, 0.0, 20.0)),  # exponential penalty for severe lateness
        np.where(
            rel_slack <= 0.0,
            1.0 + (-rel_slack) * 2.0,  # linear ramp from 0 to 1.0 as slack goes from 0 to -0.5
            1.0 / (1.0 + np.clip(rel_slack * 2.0, 0.0, 1000.0))  # soft decay for positive slack
        )
    )

    # Critical-path-aware normalization: use normalized upward_rank to avoid shallow-DAG bias
    max_ur = np.max(upward_rank) + eps
    norm_upward_rank = upward_rank / max_ur
    crit_path_work = np.maximum(remaining_work * norm_upward_rank, eps)

    # Uncertainty-coupled energy density: sqrt(uncertainty) dampens denominator to preserve low-energy viability under high risk
    energy_density = min_incremental_energy / (crit_path_work * (1.0 + np.sqrt(uncertainty) + eps))
    energy_density = np.clip(energy_density, eps, 1e12)
    energy_density_norm = robust_normalize(energy_density)
    energy_efficiency_score = -energy_density_norm

    # Uncertainty-amplified slack: multiplicative only when slack is negative (risk-triggered)
    unc_amplified_slack = np.where(
        slack < 0.0,
        slack_urgency * (1.0 + np.clip(uncertainty, 0.0, 5.0)),
        slack_urgency
    )

    # Risk-conditioned starvation guard: activates on deadline violation OR high uncertainty + long wait
    p90_wait = np.percentile(ready_wait_time, 90, method='midpoint') if N > 1 else np.max(ready_wait_time)
    starvation_mask = (slack <= 0.0) | ((uncertainty > 0.8) & (ready_wait_time > p90_wait + eps))
    starvation_guard = np.where(starvation_mask, robust_normalize(ready_wait_time), 0.0)

    # Duration penalty only for non-critical-path tasks, using robust-normalized duration
    median_ur = np.median(upward_rank) + eps
    duration_penalty = np.where(
        upward_rank <= median_ur,
        robust_normalize(task_duration),
        0.0
    )

    # Final weighted score — increased weight on urgency (DDL hard constraint), reduced on duration (secondary)
    score = (
        3.2 * unc_amplified_slack +
        1.7 * energy_efficiency_score +
        0.15 * duration_penalty +
        0.12 * starvation_guard
    )

    # Final sanitization: ensure finite, deterministic, shape-compliant output
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    return score
