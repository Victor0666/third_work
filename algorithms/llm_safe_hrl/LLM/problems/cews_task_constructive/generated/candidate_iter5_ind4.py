import numpy as np

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
    Self-evolved priority rule: deadline-hardness first, risk-calibrated criticality-energy synergy,
    starvation-aware wait gating, uncertainty-modulated scaling, and slack-proportional work penalty.
    
    Key improvements:
      - Combines Parent 2's adaptive urgency threshold (30th percentile slack) with Parent 1's smooth sigmoid slack gating
        for robust near-deadline sensitivity and hard-DDL compliance.
      - Uses unified joint robust scaling (Parent 1) over critical features to prevent noise amplification,
        but with Parent 2's sign-preserving median centering for physical interpretability.
      - Integrates Parent 2's uncertainty-amplified deadline penalty and slack-scaled ce_ratio,
        enhanced with Parent 1's criticality-energy synergy term (upward_rank * energy / duration).
      - Replaces fixed wait thresholds with dynamic wait-age gating relative to median wait time,
        avoiding brittle max-based normalization in sparse scenarios.
      - Introduces slack-resilient work penalty: penalizes heavy subtrees only when slack is depleted,
        using smooth sigmoid decay instead of hard max(0, 1-slack/ratio).
      - All operations epsilon-protected, NaN/inf-cleared, and bounded to finite ranges.
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

    # Compute base derived features safely
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_sec = np.clip(min_incremental_energy / duration, -1e6, 1e6)
    crit_energy_synergy = np.clip(upward_rank * min_incremental_energy / (duration + eps), -1e6, 1e6)

    # Unified joint feature set for robust scaling: preserves physical directionality
    joint_features = np.vstack([
        np.abs(upward_rank),
        np.abs(remaining_work),
        duration,
        np.abs(energy_per_sec),
        np.abs(crit_energy_synergy)
    ]).T
    if joint_features.size == 0:
        joint_scale = eps
    else:
        med = np.median(joint_features.flatten())
        x_centered = joint_features.flatten() - med
        q1, q3 = np.quantile(x_centered, [0.25, 0.75], method='midpoint')
        iqr = q3 - q1
        if iqr < eps:
            mad = np.median(np.abs(x_centered))
            joint_scale = mad if mad > eps else np.mean(np.abs(x_centered)) + eps
        else:
            joint_scale = iqr + eps

    def robust_joint_normalize(x):
        x_centered = x - np.median(x)
        return np.clip(x_centered / joint_scale + np.median(x), -1e6, 1e6)

    # Deadline urgency: smooth sigmoid + adaptive threshold + uncertainty amplification
    slack_sorted = np.sort(slack)
    urgency_thresh = slack_sorted[max(0, int(0.3 * len(slack_sorted)))] if N > 0 else 0.0
    sigmoid_urgency = 1.0 / (1.0 + np.exp(-slack / (np.abs(np.median(slack)) + eps)))
    is_urgent = slack <= urgency_thresh
    neg_slack = np.maximum(-slack, 0.0)
    deadline_penalty = neg_slack * (5.0 + 3.0 * np.clip(uncertainty, 0.0, 2.0))
    # Boost penalty for urgent late tasks beyond threshold
    deadline_penalty = np.where(is_urgent & (slack < 0), deadline_penalty + 8.0, deadline_penalty)

    # Criticality-energy synergy: slack-scaled and uncertainty-damped
    ce_ratio = upward_rank / (min_incremental_energy + eps)
    slack_decay = np.exp(-np.clip(slack, -100.0, 100.0) / (np.abs(np.median(slack)) + eps))
    unc_damp = 1.0 / (1.0 + np.clip(uncertainty, 0.0, 10.0))
    ce_score = -robust_joint_normalize(ce_ratio) * slack_decay * unc_damp

    # Energy score: uncertainty-weighted and normalized
    energy_base = min_incremental_energy * (1.0 + 0.6 * np.clip(uncertainty, 0.0, 1.0))
    energy_score = robust_joint_normalize(energy_base)

    # Latency score: total duration, normalized
    latency_score = robust_joint_normalize(duration)

    # Starvation-robust wait boost: activated only for tasks older than median wait time
    median_wait = np.median(ready_wait_time) if N > 0 else eps
    wait_activation = ready_wait_time > median_wait
    wait_boost = np.where(wait_activation, 0.9 * (ready_wait_time / (median_wait + eps)), 0.1 * (ready_wait_time / (median_wait + eps)))
    wait_score = -wait_boost

    # Slack-resilient work penalty: smooth penalty that activates only as slack depletes
    slack_ratio = np.clip(slack / (np.abs(np.median(slack)) + eps), -10.0, 10.0)
    # Sigmoid penalty: near zero when slack is abundant, rises smoothly as slack approaches zero
    work_penalty_factor = 1.0 / (1.0 + np.exp(-(1.0 - slack_ratio)))  # ~0 when slack_ratio >> 1, ~1 when slack_ratio <= 0
    work_norm = robust_joint_normalize(remaining_work)
    work_penalty = work_norm * work_penalty_factor

    # Final weighted combination — prioritize deadline penalty most strongly
    score = (
        1.5 * deadline_penalty +
        0.25 * latency_score +
        0.3 * energy_score +
        0.4 * ce_score +
        0.1 * wait_score +
        0.15 * work_penalty
    )

    # Ensure finite, deterministic output
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    return score
