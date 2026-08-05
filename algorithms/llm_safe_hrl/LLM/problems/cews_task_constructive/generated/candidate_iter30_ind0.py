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
    v3 priority rule: Deadline-resilient urgency + criticality-aware energy gating +
                      uncertainty-robust fairness + work-efficiency regularization +
                      slack-proximity adaptive normalization + numerical hardening.

    Key self-evolution improvements:
    - Replace fixed lateness penalty with *adaptive violation penalty*: scales logarithmically
      with |slack| to avoid dominance over other terms and preserve gradient stability.
    - Introduce *criticality-aware energy gating* using sigmoid(upward_rank) instead of linear
      thresholding, enabling smooth transition from deadline-first to energy-aware scheduling.
    - Enhance fairness with *uncertainty-relative wait boost*: normalize wait_time by median duration
      and scale only when uncertainty exceeds percentile-75, improving robustness to outliers.
    - Add *communication efficiency penalty*: explicitly penalize high min_comm_time/min_exec_time ratio
      with saturation to prevent energy waste from excessive data movement.
    - Use *slack-adaptive normalization bounds*: dynamically tighten minmax percentiles (1%-99% for tight slack,
      5%-95% otherwise) to improve discriminability near deadlines.
    - Hardened numerics: replace all np.percentile calls with deterministic fallbacks for small N,
      enforce finite-only outputs via strict clipping and NaN/inf sanitization.
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

    def robust_minmax_norm(x, tight_mode=False):
        if x.size == 0:
            return np.zeros_like(x)
        if N == 1:
            return np.zeros_like(x)
        # Adaptive percentile bounds: tighter for deadline-critical tasks
        low_p = 1.0 if tight_mode else 5.0
        high_p = 99.0 if tight_mode else 95.0
        try:
            p_low = np.percentile(x, low_p, method='lower')
            p_high = np.percentile(x, high_p, method='higher')
        except:
            p_low, p_high = np.min(x), np.max(x)
        x_clipped = np.clip(x, p_low, p_high)
        x_min = np.min(x_clipped)
        x_max = np.max(x_clipped)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        return (x_clipped - x_min) / (x_max - x_min + eps)

    duration = min_exec_time + min_comm_time + eps
    rel_slack = np.divide(slack, duration, out=np.zeros_like(slack), where=duration != 0)
    rel_slack = np.nan_to_num(rel_slack, nan=0.0, posinf=0.0, neginf=0.0)

    # Adaptive lateness penalty: log-scaled to avoid numeric dominance
    lateness_mask = slack <= 0.0
    abs_slack = np.abs(slack)
    lateness_penalty = np.where(
        lateness_mask,
        -1e9 * (1.0 + np.log1p(abs_slack + eps)),
        0.0
    )

    # Smooth urgency bias with dynamic norm mode for tight slack
    urgency_bias_raw = np.clip(-rel_slack, 0.0, 0.5)
    tight_slack_mask = (rel_slack < 0.1) & (rel_slack > -0.1)
    norm_urgency_bias = robust_minmax_norm(urgency_bias_raw, tight_mode=np.any(tight_slack_mask))

    # Energy density with slack-aware rescaling
    energy_density = np.divide(min_incremental_energy, duration + eps, out=np.zeros_like(min_incremental_energy), where=duration + eps != 0)
    energy_density = np.nan_to_num(energy_density, nan=0.0, posinf=0.0, neginf=0.0)
    slack_rescale = 1.0 + np.sqrt(np.abs(rel_slack) + eps)
    risk_adjusted_energy = energy_density / (slack_rescale + eps)
    norm_risk_energy = robust_minmax_norm(risk_adjusted_energy, tight_mode=np.any(tight_slack_mask))

    # Criticality gating via sigmoid: smooth, bounded, monotonic
    norm_upward_rank = robust_minmax_norm(upward_rank)
    criticality_gate = 1.0 / (1.0 + np.exp(-2.0 * (norm_upward_rank - 0.5)))
    energy_penalty_weight = criticality_gate * (slack > 0.0).astype(np.float64)

    # Communication efficiency penalty: saturating ratio penalty
    comm_to_exec_ratio = np.divide(min_comm_time, min_exec_time + eps, out=np.zeros_like(min_comm_time), where=min_exec_time + eps != 0)
    comm_to_exec_ratio = np.nan_to_num(comm_to_exec_ratio, nan=0.0, posinf=0.0, neginf=0.0)
    comm_eff_penalty = np.clip(comm_to_exec_ratio, 0.0, 2.0)  # cap extreme ratios

    # Work density and intensity
    work_density = np.divide(remaining_work, duration + eps, out=np.zeros_like(remaining_work), where=duration + eps != 0)
    work_density = np.nan_to_num(work_density, nan=0.0, posinf=0.0, neginf=0.0)
    norm_work_density = robust_minmax_norm(work_density)
    work_intensity_penalty = 0.7 * (1.0 - norm_work_density) + 0.3 * comm_eff_penalty

    # Uncertainty-robust fairness: relative wait boost only under high uncertainty
    wait_ratio = np.divide(ready_wait_time, duration + eps, out=np.zeros_like(ready_wait_time), where=duration + eps != 0)
    wait_ratio = np.nan_to_num(wait_ratio, nan=0.0, posinf=0.0, neginf=0.0)
    unc_p75 = np.percentile(uncertainty, 75.0) if N > 1 else np.max(uncertainty)
    fairness_boost_mask = (wait_ratio > 1.5) & (uncertainty > unc_p75)
    # Normalize wait time relative to median duration to suppress spurious boosts
    median_dur = np.median(duration) if N > 1 else duration[0]
    rel_wait = np.divide(ready_wait_time, median_dur + eps, out=np.zeros_like(ready_wait_time), where=median_dur + eps != 0)
    rel_wait = np.nan_to_num(rel_wait, nan=0.0, posinf=0.0, neginf=0.0)
    norm_rel_wait = robust_minmax_norm(rel_wait)
    fairness_boost = norm_rel_wait * fairness_boost_mask.astype(np.float64)

    # Slack-proximity gating for energy preference: strongest in [0.15, 0.75]
    slack_gate_center = 0.45
    slack_gate_width = 0.3
    slack_dist = np.abs(rel_slack - slack_gate_center)
    slack_gate = np.clip(1.0 - (slack_dist / (slack_gate_width + eps)), 0.0, 1.0)
    slack_gate = np.where((rel_slack >= 0.15) & (rel_slack <= 0.75), slack_gate, 0.0)

    # Remaining work and upward rank normalization
    norm_remaining_work = robust_minmax_norm(remaining_work, tight_mode=np.any(tight_slack_mask))
    norm_upward_rank_final = robust_minmax_norm(upward_rank, tight_mode=np.any(tight_slack_mask))

    # Final weighted score: balanced, bounded, and deadline-resilient
    score = (
        0.32 * norm_urgency_bias +
        0.26 * norm_risk_energy * energy_penalty_weight * slack_gate +
        0.15 * work_intensity_penalty +
        0.13 * (1.0 - fairness_boost) +
        0.09 * norm_remaining_work +
        0.05 * norm_upward_rank_final
    )

    # Apply lateness penalty last to dominate under violation
    score = lateness_penalty + score

    # Strict sanitization
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    score = np.clip(score, -1e12, 1e12)

    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
