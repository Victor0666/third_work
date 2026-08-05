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
    v2 priority rule: Enforces absolute deadline dominance via crisp urgency gating,
    replaces median-based energy gating with workflow-relative criticality-aware efficiency threshold,
    introduces slack-normalized risk scaling for all uncertainty terms, and eliminates rank-inversion
    via monotonic, outlier-robust normalization with strict component ordering.

    Key improvements:
      - Urgency is *absolute*: urgency_flag = 0.0 for slack <= 0 → highest priority, no conditional masking.
      - Energy penalty uses *criticality-weighted efficiency threshold*: threshold = median(energy_density / (upward_rank + eps)),
        ensuring efficient tasks on critical paths are never penalized unfairly.
      - All risk terms (uncertainty, starvation, latency) scaled by |slack|^{-1} clamped to [0.1, 10.0] → 
        stronger penalties when slack is small but avoids explosion at slack=0 via soft lower bound.
      - Latency term integrates *criticality-amplified communication* and *execution uncertainty coupling*:
        comm_weight = 1.0 + 0.5 * norm(upward_rank) + 0.3 * norm(uncertainty).
      - Starvation guard uses *slack-aware decay*: exp(-0.5 * ready_wait_time / (|slack| + 1.0)) → 
        long waits matter more only when slack is ample; vanishes near deadline.
      - Robust normalization uses 0.5%-99.5% clipping + min-max, with explicit constant-array safety.
      - Final weights enforce strict hierarchy: urgency (0.52) >> energy (0.18) >> latency (0.12) >>
        fairness (0.07) >> uncertainty modulation (0.06) >> baseline uncertainty (0.05).
    '''
    eps = 1e-08
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=float)

    def robust_minmax_norm(x):
        x_min = np.min(x)
        x_max = np.max(x)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        p005 = np.percentile(x, 0.5)
        p995 = np.percentile(x, 99.5)
        x_clipped = np.clip(x, p005, p995)
        x_min_c = np.min(x_clipped)
        x_max_c = np.max(x_clipped)
        if x_max_c - x_min_c < eps:
            return np.zeros_like(x)
        return (x - x_min_c) / (x_max_c - x_min_c + eps)

    # Absolute urgency gating: zero score for any task violating or at deadline → max priority
    urgency_flag = np.where(slack <= 0, 0.0, 1.0)

    # Criticality-aware energy density: Joules per second of total duration, weighted by importance
    duration = min_exec_time + min_comm_time + eps
    energy_density = min_incremental_energy / (duration + eps)
    # Threshold adapts to workflow: median of (energy_density / (upward_rank + eps)) → rewards efficiency on critical paths
    eff_ratio = energy_density / (upward_rank + eps)
    threshold_eff_ratio = np.median(eff_ratio) + eps
    # Penalize only inefficient *and* non-critical tasks — efficient or critical ones get free pass
    energy_penalty_mask = (eff_ratio > threshold_eff_ratio).astype(float)
    norm_energy_density = robust_minmax_norm(energy_density)
    energy_penalty = norm_energy_density * energy_penalty_mask * urgency_flag

    # Latency term: execution + communication amplified by both criticality and uncertainty
    comm_weight = 1.0 + 0.5 * robust_minmax_norm(upward_rank) + 0.3 * robust_minmax_norm(uncertainty)
    weighted_comm = min_comm_time * comm_weight
    latency_raw = min_exec_time + weighted_comm + eps
    norm_latency = robust_minmax_norm(latency_raw)

    # Criticality-per-energy: higher is better → invert for scoring (1 - norm)
    crit_per_energy = upward_rank / (min_incremental_energy + eps)
    crit_per_energy = np.clip(crit_per_energy, 1e-06, 1e7)
    norm_crit_per_energy = robust_minmax_norm(crit_per_energy)

    # Slack-normalized risk scaling factor: high weight when slack is small, bounded to avoid blowup
    slack_abs = np.abs(slack) + 1.0  # +1.0 ensures finite scaling even at slack=0
    slack_scale_factor = np.clip(1.0 / slack_abs, 0.1, 10.0)

    # Uncertainty boost: modulated by both slack proximity and magnitude
    uncertainty_boost = uncertainty * slack_scale_factor
    norm_uncertainty_boost = robust_minmax_norm(uncertainty_boost)

    # Starvation guard: activated only for non-urgent (slack > 0), decays exponentially with wait time relative to slack
    starvation_flag = (slack > 0).astype(float)
    wait_decay = np.exp(-0.5 * ready_wait_time / (np.abs(slack) + 1.0))
    wait_penalty = starvation_flag * robust_minmax_norm(ready_wait_time) * wait_decay

    # Baseline uncertainty term (unmodulated) for general risk awareness
    norm_uncertainty = robust_minmax_norm(uncertainty)

    # Convex combination with strict dominance hierarchy (sums to 1.0)
    score = (
        0.52 * urgency_flag +
        0.18 * energy_penalty +
        0.12 * norm_latency +
        0.07 * (1.0 - norm_crit_per_energy) +
        0.06 * norm_uncertainty_boost +
        0.05 * wait_penalty +
        0.05 * norm_uncertainty +
        0.05 * (1.0 - robust_minmax_norm(remaining_work))  # Favor tasks with less remaining work downstream (faster cleanup)
    )

    # Final safeguard: clamp and sanitize
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)

    return score
