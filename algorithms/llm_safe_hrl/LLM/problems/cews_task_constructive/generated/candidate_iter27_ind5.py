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
    v5 priority rule: Monotonic urgency-preserving hybrid optimizer with:
      - Hard urgency dominance: all slack<=0 tasks get fixed ultra-low score (no gating)
      - Criticality-energy coupling via *slack-normalized* energy density, not sigmoid modulation
      - Starvation rescue strictly gated by positive slack AND high relative wait AND low work density
      - Residual work term replaced by *normalized remaining work per unit criticality* to avoid deadline dilution
      - Uncertainty boost simplified: uses dur_uncertainty * exp(-max(0,-slack)/tau) for late-task risk awareness
      - All normalizations use robust minmax with explicit degeneracy fallbacks; no percentile when N<3
      - Final weights rebalanced: urgency (0.47), critical latency (0.21), energy (0.16), fairness (0.08),
        starvation (0.05), uncertainty (0.03) — prioritizing deadline fidelity and energy efficiency
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

    def robust_minmax_norm(x):
        if x.size == 0:
            return np.zeros_like(x)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x_finite = x[np.isfinite(x)]
        if x_finite.size == 0:
            return np.zeros_like(x)
        if x_finite.size < 3:
            x_min, x_max = (np.min(x_finite), np.max(x_finite))
        else:
            p01 = np.percentile(x_finite, 1.0, method='midpoint')
            p99 = np.percentile(x_finite, 99.0, method='midpoint')
            x_clipped = np.clip(x_finite, p01, p99)
            x_min, x_max = (np.min(x_clipped), np.max(x_clipped))
        if x_max - x_min < eps:
            return np.zeros_like(x)
        normed = np.zeros_like(x)
        valid_mask = np.isfinite(x)
        normed[valid_mask] = (x[valid_mask] - x_min) / (x_max - x_min + eps)
        return normed

    # Compute base duration and risk-adjusted duration
    duration = min_exec_time + min_comm_time + eps
    risk_adjusted_duration = duration * (1.0 + uncertainty)

    # Energy density: risk-adjusted, prevents inversion under volatility
    energy_density = np.divide(min_incremental_energy, risk_adjusted_duration, out=np.zeros_like(min_incremental_energy), where=risk_adjusted_duration != 0)
    energy_density = np.nan_to_num(energy_density, nan=0.0, posinf=0.0, neginf=0.0)

    # Urgency flag: strict hard constraint — all overdue/urgent tasks dominate
    is_urgent = (slack <= 0.0).astype(float)

    # Adaptive slack threshold only for non-urgent tasks (positive slack only)
    positive_slack = slack[slack > 0]
    adaptive_slack_thresh = np.median(positive_slack) if len(positive_slack) > 0 else 1.0

    # Tight slack mask: only tasks with slack <= threshold are considered for energy/criticality penalties
    tight_slack_mask = (slack <= adaptive_slack_thresh).astype(float)

    # Upward rank normalization
    norm_upward_rank = robust_minmax_norm(upward_rank)

    # Critical latency: duration weighted by normalized rank — preserves monotonicity
    critical_latency_raw = duration * (1.0 + 0.5 * norm_upward_rank)
    norm_critical_latency = robust_minmax_norm(critical_latency_raw)

    # Normalize energy density
    norm_energy_density = robust_minmax_norm(energy_density)

    # Energy significance mask: only penalize meaningful contributors (> p10 of finite values)
    energy_density_finite = energy_density[np.isfinite(energy_density)]
    energy_floor = np.percentile(energy_density_finite, 10) + eps if len(energy_density_finite) > 0 else eps
    energy_significant_mask = (energy_density >= energy_floor).astype(float)

    # High-rank mask: avoids penalizing low-importance tasks
    high_rank_mask = (upward_rank >= np.median(upward_rank) + eps).astype(float) if N > 1 else np.ones(N)

    # Energy penalty: only applied to tight-slack, high-rank, significant-energy tasks
    energy_penalty = norm_energy_density * tight_slack_mask * high_rank_mask * energy_significant_mask

    # Fairness: wait-per-work ratio, normalized and gated only for non-urgent tasks
    wait_per_work = np.divide(ready_wait_time, remaining_work + eps, out=np.zeros_like(ready_wait_time), where=remaining_work + eps != 0)
    wait_per_work = np.nan_to_num(wait_per_work, nan=0.0, posinf=0.0, neginf=0.0)
    norm_wait_per_work = robust_minmax_norm(wait_per_work)
    wpw_finite = wait_per_work[np.isfinite(wait_per_work)]
    wait_threshold = np.median(wpw_finite) * 1.5 + eps if len(wpw_finite) > 0 else eps
    wait_gate = (wait_per_work >= wait_threshold).astype(float)
    wait_penalty = (1.0 - is_urgent) * norm_wait_per_work * wait_gate

    # Starvation rescue: only for tasks with positive slack, high wait_ratio, AND low work density (small tasks starved)
    wait_ratio = np.divide(ready_wait_time, duration + eps, out=np.zeros_like(ready_wait_time), where=duration + eps != 0)
    wait_ratio = np.nan_to_num(wait_ratio, nan=0.0, posinf=0.0, neginf=0.0)
    ratio_finite = wait_ratio[np.isfinite(wait_ratio)]
    median_ratio = np.median(ratio_finite) if len(ratio_finite) > 0 else 1.0
    # Starvation requires: positive slack, high relative wait, AND low remaining_work per unit rank (i.e., small but critical)
    work_density = np.divide(remaining_work, (upward_rank + eps), out=np.zeros_like(remaining_work), where=upward_rank + eps != 0)
    work_density = np.nan_to_num(work_density, nan=0.0, posinf=0.0, neginf=0.0)
    is_starvable = (
        (wait_ratio > 1.5 * median_ratio) & 
        (slack > 0.0) & 
        (work_density <= np.median(work_density[np.isfinite(work_density)]) + eps if np.any(np.isfinite(work_density)) else True)
    )
    starvation_boost = np.where(is_starvable, wait_ratio * (1.0 + 0.1 * norm_upward_rank), 0.0)
    norm_starvation = robust_minmax_norm(starvation_boost)

    # Uncertainty boost: sensitive to both duration uncertainty and negative slack (lateness risk)
    dur_uncertainty = np.divide(uncertainty, duration, out=np.zeros_like(uncertainty), where=duration != 0)
    dur_uncertainty = np.nan_to_num(dur_uncertainty, nan=0.0, posinf=0.0, neginf=0.0)
    # Boost increases for overdue tasks: exp(-max(0,-slack)/tau) → larger when slack is more negative
    tau = 15.0
    lateness_bias = np.exp(-np.maximum(0.0, -slack) / tau)  # peaks at slack=0, decays for positive slack
    uncertainty_boost = np.clip(dur_uncertainty * lateness_bias, 0.0, 1.0)
    norm_uncertainty_boost = robust_minmax_norm(uncertainty_boost)

    # Residual work term replaced by *critical work density*: remaining_work / (upward_rank + eps), normalized
    # This favors scheduling small-but-critical tasks early, avoiding deadline erosion from large low-rank work
    critical_work_density = np.divide(remaining_work, (upward_rank + eps), out=np.zeros_like(remaining_work), where=upward_rank + eps != 0)
    critical_work_density = np.nan_to_num(critical_work_density, nan=0.0, posinf=0.0, neginf=0.0)
    norm_critical_work_density = robust_minmax_norm(critical_work_density)

    # Combine contributions with monotonic urgency dominance
    non_urgent_contrib = (
        0.21 * norm_critical_latency +
        0.16 * energy_penalty +
        0.08 * wait_penalty +
        0.05 * norm_starvation +
        0.03 * norm_uncertainty_boost +
        0.03 * norm_critical_work_density
    )

    # Base score for non-urgent tasks; urgent tasks get fixed ultra-low score
    score = np.where(is_urgent, -1000000000000.0, 0.47 + non_urgent_contrib)

    # Clip and sanitize
    score = np.clip(score, -1000000000000.0, 1000000000000.0)
    score = np.nan_to_num(score, nan=1000000000000.0, posinf=1000000000000.0, neginf=-1000000000000.0)

    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
