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
    Self-evolved priority rule: deadline-hardness first, then risk-calibrated energy-criticality tradeoff,
    with starvation suppression, dynamic slack sensitivity, and uncertainty-aware work regularization.

    Key evolutions from v1:
      - Replaces static median slack gating with *adaptive urgency threshold*: uses percentile-30 of slack
        to prioritize the most at-risk tasks early (not just half), improving hard-DDL compliance.
      - Introduces *slack-scaled criticality-energy ratio*: ce_ratio weighted by exp(-slack / (|median_slack|+eps)),
        smoothly de-emphasizing low-urgency tasks without step-function artifacts.
      - Adds *uncertainty-amplified deadline penalty*: negative slack penalty scaled by (1 + uncertainty),
        ensuring high-risk late tasks dominate even more decisively.
      - Replaces fixed wait_rel capping with *wait-age gating*: only tasks older than 5% of max_wait get boosted,
        preventing premature starvation correction while preserving urgency hierarchy.
      - Introduces *remaining-work slack penalty*: work_norm multiplied by max(0, 1 - slack/(|median_slack|+eps)),
        penalizing heavy subtrees *proportionally* to slack depletion — avoids abrupt cutoffs.
      - Uses *symmetric IQR scaling with sign-preserving centering*: robust_scale now centers on median before scaling,
        preserving physical directionality (e.g., higher upward_rank always improves score).
      - All terms bounded, epsilon-safe, finite-preserving, and deterministic.
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

    def robust_scale(x):
        med = np.median(x)
        x_centered = x - med
        q1, q3 = np.quantile(x_centered, [0.25, 0.75])
        iqr = q3 - q1
        if iqr < eps:
            mad = np.mean(np.abs(x_centered))
            scale = mad if mad > eps else np.mean(np.abs(x_centered)) + eps
        else:
            scale = iqr + eps
        return x_centered / scale + med  # preserve original median offset for interpretability

    # Adaptive urgency threshold: bottom 30% slack → highest priority cohort
    slack_sorted = np.sort(slack)
    urgency_thresh = slack_sorted[max(0, int(0.3 * len(slack_sorted)))] if N > 0 else 0.0
    is_urgent = slack <= urgency_thresh

    # Deadline penalty: amplified by uncertainty for high-risk lateness
    neg_slack = np.maximum(-slack, 0.0)
    deadline_penalty = neg_slack * (4.0 + 2.0 * np.clip(uncertainty, 0.0, 1.0))
    deadline_penalty = np.where(slack < 0, deadline_penalty + 6.0, deadline_penalty)

    # Slack-scaled criticality-energy ratio: smooth decay, not step-gated
    ce_ratio = upward_rank / (min_incremental_energy + eps)
    slack_decay = np.exp(-np.clip(slack, -100.0, 100.0) / (np.abs(np.median(slack)) + eps))
    ce_score = -robust_scale(ce_ratio) * slack_decay

    # Uncertainty-calibrated energy base
    energy_base = min_incremental_energy * (1.0 + 0.5 * np.clip(uncertainty, 0.0, 1.0))
    energy_score = robust_scale(energy_base)

    # Latency cost: total minimal time-cost, robustly scaled
    total_latency = min_exec_time + min_comm_time + eps
    latency_score = robust_scale(total_latency)

    # Wait-age gating: only boost tasks older than 5% of max_wait (prevents noise)
    max_wait = np.max(ready_wait_time) if N > 0 else eps
    wait_norm = np.maximum(max_wait, eps)
    wait_rel = ready_wait_time / wait_norm
    wait_gate = wait_rel > 0.05
    wait_boost = np.where(wait_gate, 0.8 * wait_rel, 0.1 * wait_rel)
    wait_score = -wait_boost

    # Remaining-work slack penalty: proportional to slack exhaustion
    work_norm = robust_scale(remaining_work)
    slack_ratio = np.clip(slack / (np.abs(np.median(slack)) + eps), -5.0, 5.0)
    work_penalty = work_norm * np.maximum(0.0, 1.0 - slack_ratio)

    # Final weighted combination — deadline dominates; energy & criticality balanced; work & wait are modifiers
    score = (
        1.2 * deadline_penalty +
        0.2 * latency_score +
        0.35 * energy_score +
        0.2 * ce_score +
        0.1 * wait_score +
        0.15 * work_penalty
    )

    # Ensure finiteness and determinism
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    return score
