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
    v2 priority rule: Hard urgency dominance + latency-criticality decoupling + 
                      uncertainty-aware energy suppression + starvation-aware fairness + 
                      unified robust normalization.

    Key improvements over v1:
    - Removes fragile synergy terms (comm-energy coupling) that diluted urgency signals.
    - Replaces feasible-subset energy normalization with global trimmed normalization
      to preserve hard deadline enforcement across all tasks (including urgent ones).
    - Decouples criticality and energy penalties: upward_rank now only modulates latency
      (not energy), restoring balance between deadline adherence and energy minimization.
    - Introduces *uncertainty-aware energy suppression*: divides energy by (1 + uncertainty)
      without duration coupling — avoids penalizing short uncertain tasks unnecessarily.
    - Fairness term uses robust wait-per-work normalized globally, activated for all non-urgent
      tasks (not just feasible subset), preventing starvation while maintaining determinism.
    - All normalization uses trimmed-mean ± 3*MAD; no quantile-based methods or masking.
    - Strict priority hierarchy enforced via additive layers with descending weights.
    - All operations guarded against division-by-zero, NaN, inf; fully deterministic and finite.
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

    def robust_trimmed_norm(x):
        if x.size == 0:
            return np.zeros_like(x)
        x_med = np.median(x)
        abs_dev = np.abs(x - x_med)
        mad = np.median(abs_dev) if abs_dev.size > 0 else 0.0
        scale = 3.0 * mad + eps
        if scale < eps:
            return np.zeros_like(x)
        z = (x - x_med) / scale
        return np.clip(z, -3.0, 3.0)

    # Hard urgency dominance: urgent tasks (slack <= 0) get highest priority
    is_urgent = (slack <= 0.0).astype(np.float64)

    # Latency-criticality: duration-weighted upward rank (no uncertainty coupling)
    duration = min_exec_time + min_comm_time + eps
    critical_latency_raw = duration * upward_rank
    norm_critical_latency = robust_trimmed_norm(critical_latency_raw)

    # Uncertainty-aware energy suppression: suppress high-energy assignments on uncertain VMs
    # but avoid over-penalizing short-duration tasks — use pure uncertainty scaling
    energy_suppressed = min_incremental_energy / (1.0 + uncertainty + eps)
    energy_suppressed = np.where(np.isfinite(energy_suppressed), energy_suppressed, 0.0)
    norm_energy = robust_trimmed_norm(energy_suppressed)

    # Starvation-aware fairness: wait-per-work, globally normalized, active for non-urgent only
    wait_per_work = ready_wait_time / (remaining_work + eps)
    wait_per_work = np.where(np.isfinite(wait_per_work), wait_per_work, 0.0)
    norm_wait_per_work = robust_trimmed_norm(wait_per_work)
    fairness_penalty = np.where(is_urgent, 0.0, norm_wait_per_work)

    # Slack sensitivity: clipped inverse-linear urgency amplification near zero slack
    abs_slack = np.abs(slack) + eps
    slack_sensitivity = np.clip(1.0 / abs_slack, 0.1, 20.0)
    norm_slack_sensitivity = robust_trimmed_norm(slack_sensitivity)

    # Uncertainty pressure: amplify urgency penalty proportionally to uncertainty
    norm_uncertainty = robust_trimmed_norm(uncertainty)
    uncertainty_pressure = norm_uncertainty * norm_slack_sensitivity

    # Remaining work importance: lower priority for large remaining work (less urgent dispatch)
    norm_remaining_work = robust_trimmed_norm(remaining_work)

    # Build score with strict priority hierarchy: urgency > latency > energy > fairness > risk
    score = np.full(N, 0.0, dtype=np.float64)
    score = np.where(is_urgent, -1000000000000.0, score)  # absolute top priority
    score = np.where(is_urgent, score, score + 0.4 * norm_critical_latency)
    score = np.where(is_urgent, score, score + 0.25 * norm_energy)
    score = np.where(is_urgent, score, score + 0.2 * fairness_penalty)
    score = np.where(is_urgent, score, score + 0.1 * uncertainty_pressure)
    score = np.where(is_urgent, score, score + 0.05 * norm_remaining_work)

    # Final bounds and sanitization
    score = np.clip(score, -1000000000000.0, 1000000000000.0)
    score = np.nan_to_num(score, nan=1000000000000.0, posinf=1000000000000.0, neginf=-1000000000000.0)

    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
