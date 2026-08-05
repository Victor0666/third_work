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

    Key improvements over v1:
      - Restores robust min-max normalization (stable at N=1, avoids outlier sensitivity)
      - Replaces unstable criticality-energy synergy with *slack-gated criticality ratio*:
        upward_rank / (min_incremental_energy + eps) * (1 + clamp(-slack/median_duration, 0, 1))
        — preserves critical-path focus while safely penalizing late tasks.
      - Uncertainty now *only* scales duration (not energy reversal), avoiding non-monotonic tradeoffs.
      - Starvation guard uses *work-normalized wait pressure*: ready_wait_time / (remaining_work + eps),
        gated by slack > 0 AND low latency — eliminates density heuristic fragility.
      - Energy reversal is purely sign-consistent and uncertainty-agnostic: norm_energy if slack > 0, else 1-norm_energy.
      - All components bounded, finite, deterministic, and shape-(N,) guaranteed.
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
    
    # Base urgency: linear slack gap normalized to [0,1], clipped and reversed
    slack_gap = np.clip(median_slack - slack, 0.0, None)  # only penalize slack deficit
    base_urgency = robust_minmax_norm(slack_gap + eps)

    # Criticality-energy coupling: upward_rank per unit energy, boosted only when slack is negative
    # Avoids noise amplification at low energy; maintains monotonicity
    crit_energy_ratio = upward_rank / (min_incremental_energy + eps)
    # Boost only for urgent tasks: add slack-deficit factor (0 to 1)
    deficit_factor = np.clip(-slack / (median_duration + eps), 0.0, 1.0)
    amplified_crit_ratio = crit_energy_ratio * (1.0 + deficit_factor)
    norm_crit_ratio = robust_minmax_norm(np.clip(amplified_crit_ratio, eps, 1e6))

    # Energy fairness: pure sign-consistent reversal — no uncertainty coupling
    norm_energy = robust_minmax_norm(min_incremental_energy)
    energy_reversed = np.where(slack > 0, norm_energy, 1.0 - norm_energy)

    # Uncertainty-augmented duration as primary urgency driver (stable, monotonic)
    unc_duration = total_latency * (1.0 + uncertainty)
    norm_unc_duration = robust_minmax_norm(unc_duration)

    # Work-aware starvation control: normalized wait pressure, gated by feasibility & low-latency
    # Prevents starvation only when task is both waiting long *and* cheap to execute
    work_normalized_wait = ready_wait_time / (remaining_work + eps)
    wait_threshold = np.quantile(work_normalized_wait, 0.9, method='midpoint') + eps
    wait_pressure = np.clip(work_normalized_wait / (wait_threshold + eps), 0.0, 1.0)
    # Gate: only activate starvation guard for tasks with positive slack AND short latency
    latency_gate = (total_latency <= np.median(total_latency)).astype(float)
    slack_gate = (slack > 0).astype(float)
    starvation_term = 1.0 - wait_pressure * latency_gate * slack_gate

    # Work penalty: only applied under deadline pressure (negative slack)
    slack_sensitivity = np.exp(-np.clip(np.maximum(-slack, 0.0), 0.0, 100.0) / (median_duration + eps))
    norm_work = robust_minmax_norm(remaining_work)
    work_penalty = norm_work * slack_sensitivity

    # Final weighted score: urgency dominant, energy fairness strengthened, starvation tightly controlled
    score = (
        0.45 * base_urgency +
        0.24 * energy_reversed +
        0.13 * norm_unc_duration +
        0.09 * (1.0 - norm_crit_ratio) +
        0.06 * starvation_term +
        0.03 * work_penalty
    )

    # Ensure finiteness and shape compliance
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e9, 1e9)
    return score
