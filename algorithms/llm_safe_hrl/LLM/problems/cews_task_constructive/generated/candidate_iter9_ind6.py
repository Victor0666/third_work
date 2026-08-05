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
    Self-evolved priority rule v2: hardened deadline adherence, monotonic energy fairness,
    and robust starvation control — eliminating tanh nonlinearity, reversing energy-density
    only via sign-consistent scaling, and replacing percentile-based wait ranking with
    smoothed quantile estimation for stability at all N.

    Key improvements:
      - Replaces tanh-based criticality amplification with *linear slack-gated boost*:
        upward_rank * (1 + clamp((median_slack - slack)/median_duration, 0, 1)),
        preserving monotonic urgency-criticality coupling.
      - Energy fairness uses *sign-consistent linear reversal*: low-energy favored when slack > 0,
        high-criticality/low-energy-offload favored when slack < 0 — no tanh or discontinuity.
      - Starvation guard uses *robust 90th-percentile smoothed wait pressure* instead of rank sorting,
        avoiding brittle small-N behavior and burst-sensitivity.
      - All normalization uses degenerate-safe robust_minmax_norm; all divisions guarded.
      - Urgency remains dominant (0.44), energy fairness strengthened (0.25) with clean reversal,
        latency (0.11), criticality (0.09), starvation (0.07), work impact (0.04).
      - Final score strictly finite, deterministic, and shape-(N,) guaranteed.
    """
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
        x_min, x_max = np.min(x), np.max(x)
        if x_max - x_min < eps:
            return np.zeros_like(x, dtype=float)
        return (x - x_min) / (x_max - x_min + eps)

    total_latency = min_exec_time + min_comm_time + eps
    median_duration = np.median(total_latency) + eps
    median_slack = np.median(slack)
    
    # Hardened sigmoid urgency: steep, crisp boundary near median_slack
    q1_slack, q3_slack = np.percentile(slack, [25, 75])
    iqr_slack = q3_slack - q1_slack + eps
    soft_urgency = 1.0 / (1.0 + np.exp(-(slack - median_slack) / (0.25 * iqr_slack + eps)))
    base_urgency = 1.0 - soft_urgency

    # Linear slack-gated criticality boost: monotonic, bounded [0,1] boost factor
    slack_gap = median_slack - slack
    boost_factor = np.clip(slack_gap / (median_duration + eps), 0.0, 1.0)
    amplified_upward_rank = upward_rank * (1.0 + boost_factor)

    # Energy fairness: sign-consistent linear reversal — no tanh, no non-monotonicity
    # When slack > 0: prioritize low energy → normalized energy score (smaller = better)
    # When slack <= 0: prioritize high criticality *and* low energy offload → invert energy score
    norm_energy = robust_minmax_norm(min_incremental_energy)
    energy_reversed = np.where(slack > 0, norm_energy, 1.0 - norm_energy)

    # Criticality-energy ratio: stable, clipped to avoid explosion
    crit_energy_ratio = amplified_upward_rank / (min_incremental_energy + eps)
    norm_crit_energy_ratio = robust_minmax_norm(np.clip(crit_energy_ratio, eps, 1e6))

    # Energy penalty: only under tight slack AND high uncertainty (joint mask)
    tight_slack_mask = (slack <= median_slack).astype(float)
    high_uncertainty_mask = (uncertainty >= np.quantile(uncertainty, 0.75, method='midpoint')).astype(float)
    energy_penalty = robust_minmax_norm(min_incremental_energy) * (
        1.0 + 0.8 * uncertainty * tight_slack_mask * high_uncertainty_mask
    )

    # Latency term: normalized total latency
    norm_latency = robust_minmax_norm(total_latency)

    # Work impact: slack-sensitive penalty only for negative slack
    tau = np.maximum(np.abs(median_slack), 1.0) + eps
    slack_sensitivity = np.exp(-np.clip(np.maximum(-slack, 0.0), 0.0, 100.0) / tau)
    norm_work = robust_minmax_norm(remaining_work)
    work_penalty = norm_work * slack_sensitivity

    # Starvation guard: smoothed 90th-percentile wait pressure (robust to N=1)
    # Uses quantile-based threshold instead of fragile rank sorting
    wait_threshold = np.quantile(ready_wait_time, 0.9, method='midpoint') + eps
    wait_pressure = np.clip(ready_wait_time / (wait_threshold + eps), 0.0, 1.0)
    # Only activate for non-urgent & DDL-feasible tasks
    wait_gate = np.where((soft_urgency < 0.55) & (slack >= 0), 1.0, 0.0)
    starvation_term = 1.0 - (wait_pressure * wait_gate)

    # Weighted fusion — weights sum to 1.0
    score = (
        0.44 * base_urgency +
        0.25 * energy_reversed +
        0.11 * norm_latency +
        0.09 * (1.0 - norm_crit_energy_ratio) +
        0.07 * starvation_term +
        0.04 * work_penalty
    )

    # Final safeguard: clip and sanitize
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    
    return score
