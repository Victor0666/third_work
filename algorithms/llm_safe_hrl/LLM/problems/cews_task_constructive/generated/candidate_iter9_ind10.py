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
    Self-evolved v2 priority rule: balances deadline fidelity, energy-awareness, and robustness.
    Key improvements over v1:
    - Replaces brittle exponential penalty with *smooth quadratic slack violation cost* → avoids distortion under mild lateness.
    - Uses *normalized wait-ratio aging* (ready_wait_time / (|slack| + eps)) instead of percentile → fully deterministic, scalable to N=1.
    - Broadens uncertainty gating to (0, 3.0] as reflected; adds *uncertainty-aware slack scaling* for finer risk modulation.
    - Introduces *energy-efficiency saturation guard*: caps critical-energy density contribution when slack is very tight (< 0.5s) to prevent energy greed at expense of deadline.
    - All normalizations use safe_mad_normalize with consistent clipping; final score strictly finite & shape-(N,).
    """
    eps = 1e-08
    # Sanitize all inputs: ensure finite, replace inf/nan with safe bounds
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=eps, posinf=1e6, neginf=eps)

    def safe_mad_normalize(x):
        """Robust MAD normalization: handles N=0, N=1, constant arrays, outliers."""
        if x.size == 0:
            return np.zeros_like(x)
        if x.size == 1:
            return np.zeros_like(x)
        median_x = np.median(x)
        mad = np.median(np.abs(x - median_x)) + eps
        if mad < eps:
            return np.zeros_like(x)
        normed = (x - median_x) / mad
        return np.clip(normed, -3.0, 3.0)

    # === Deadline urgency: smooth, bounded, and adaptive ===
    # Use sigmoid base risk + quadratic violation penalty (not exponential) → no explosion for small negative slack
    abs_slack = np.abs(slack)
    slack_scale = np.median(abs_slack) + eps
    sigmoid_risk = 1.0 / (1.0 + np.exp(-slack / slack_scale))
    # Quadratic penalty for slack < 0: grows smoothly, capped at 10.0
    quad_penalty = np.where(slack < 0, (np.abs(slack) / (slack_scale + eps)) ** 2, 0.0)
    quad_penalty = np.clip(quad_penalty, 0.0, 10.0)
    deadline_score = safe_mad_normalize(sigmoid_risk + 0.4 * quad_penalty)

    # === Energy term: slack-gated & saturation-guarded ===
    # Only activate when slack > 0; suppress contribution when slack < 0.5s (deadline-critical mode)
    energy_denom = min_incremental_energy + eps
    base_density = (upward_rank + eps) * (remaining_work + eps) / energy_denom
    # Saturation guard: reduce weight when extremely tight (slack < 0.5s)
    saturation_weight = np.where(slack < 0.5, 0.3, 1.0)
    critical_energy_density = np.where(slack > 0.0, base_density * saturation_weight, 0.0)
    critical_energy_norm = safe_mad_normalize(critical_energy_density)

    # === Critical-path boost: normalized by workflow-wide median RW ===
    median_rw = np.median(remaining_work) + eps
    cp_boost = upward_rank * (remaining_work / median_rw)
    cp_boost_norm = safe_mad_normalize(cp_boost)

    # === Fairness: latency burden modulated by wait-time ratio (not percentile) ===
    total_latency = min_exec_time + min_comm_time + eps
    max_latency = np.max(total_latency) + eps
    latency_ratio = total_latency / max_latency
    # Wait-ratio aging: deterministic, works for any N including 1
    wait_ratio = ready_wait_time / (np.abs(slack) + eps)
    wait_modulator = np.tanh(0.9 * wait_ratio)  # Stronger response to waiting under pressure
    fairness_boost = np.clip(latency_ratio * (1.0 + 0.5 * wait_modulator), 0.0, 0.15)

    # === Uncertainty: broadened gating (0 < slack <= 3.0), scaled by slack proximity ===
    uncertainty_gated = np.where(
        (slack > 0.0) & (slack <= 3.0) & (uncertainty > 0.03),
        uncertainty * (1.0 / (1.0 + np.exp(-(3.0 - slack)))),
        0.0
    )
    uncertainty_norm = safe_mad_normalize(uncertainty_gated)

    # === Aging: unified into wait_ratio modulator (no separate term) → cleaner, deterministic ===
    # Already embedded in fairness_boost via wait_modulator; no redundant aging_term needed

    # Final weighted combination: higher deadline weight, guarded energy & CP terms
    score = (
        +5.2 * deadline_score
        - 2.2 * critical_energy_norm
        - 1.4 * cp_boost_norm
        + 0.13 * fairness_boost
        + 0.09 * uncertainty_norm
    )

    # Strict final sanitization: ensure finite, shape-(N,), no scalars
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    return score.reshape(-1)
