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
    Hybrid priority rule: hard deadline compliance first, then risk-adjusted energy efficiency,
    criticality-per-latency, and starvation-avoidance — with robust IQR scaling and slack-gated dynamics.
    
    Key improvements:
      - Uses *adaptive slack gating*: urgent tasks (slack < median_slack) activate deadline penalty AND 
        criticality-energy ratio boost; non-urgent use relaxed normalization.
      - Combines Parent 2's robust IQR scaling (outlier-resilient) with Parent 1's *criticality-energy ratio* 
        (upward_rank / (min_incremental_energy + eps)), but only applied under urgency to favor high-impact-low-risk work.
      - Introduces *uncertainty-calibrated energy penalty*: min_incremental_energy scaled by (1 + 0.5 * uncertainty),
        amplifying energy awareness when risk is high, without distorting low-risk regimes.
      - Replaces linear wait boosting with *relative waiting incentive*: ready_wait_time normalized by max_wait,
        capped and gated by uncertainty > 0.05 — avoids starvation while preventing dominance over deadlines.
      - Adds *remaining-work regularization*: downward weights tasks with excessive residual work if slack is tight,
        discouraging late-starting heavy subtrees when time is scarce.
      - All operations epsilon-safe, finite-preserving, deterministic, and shape-compliant.
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

    # Robust IQR-based scaling function
    def robust_scale(x):
        q1, q3 = np.quantile(x, [0.25, 0.75])
        iqr = q3 - q1
        if iqr < eps:
            mad = np.mean(np.abs(x - np.median(x)))
            scale = mad if mad > eps else np.mean(np.abs(x)) + eps
        else:
            scale = iqr + eps
        return x / scale

    # Adaptive urgency gating: tasks with slack <= median_slack are urgent
    slack_sorted = np.sort(slack)
    median_slack = np.median(slack) if N > 0 else 0.0
    is_urgent = slack <= median_slack

    # Hard deadline penalty: additive, non-multiplicative, dominates score
    deadline_penalty = np.maximum(-slack, 0.0) * 4.0
    # Extra penalty for violated deadlines
    deadline_penalty = np.where(slack < 0, deadline_penalty + 6.0, deadline_penalty)

    # Criticality-energy ratio: only activated for urgent tasks to avoid rewarding low-energy trivial work
    ce_ratio = upward_rank / (min_incremental_energy + eps)
    ce_score = np.where(is_urgent, -robust_scale(ce_ratio), -0.3 * robust_scale(ce_ratio))

    # Uncertainty-calibrated energy penalty: higher weight when uncertainty is high
    energy_base = min_incremental_energy * (1.0 + 0.5 * np.clip(uncertainty, 0.0, 1.0))
    energy_score = robust_scale(energy_base)

    # Latency components (execution + communication) — jointly scaled
    total_latency = min_exec_time + min_comm_time + eps
    latency_score = robust_scale(total_latency)

    # Relative waiting incentive: only boosted when uncertainty > 0.05, capped at 0.4 contribution
    max_wait = np.max(ready_wait_time) if N > 0 else eps
    wait_norm = np.maximum(max_wait, eps)
    wait_rel = np.clip(ready_wait_time / wait_norm, 0.0, 1.0)
    wait_score = np.where(uncertainty > 0.05, -0.8 * wait_rel, -0.2 * wait_rel)

    # Remaining-work regularization: penalize large remaining_work under tight slack
    work_norm = robust_scale(remaining_work)
    work_penalty = np.where(is_urgent, 0.25 * work_norm, 0.05 * work_norm)

    # Combine scores with prioritized weighting
    # Deadline penalty dominates; CE and energy drive tradeoff in feasible region; wait/work regularize
    score = (
        1.0 * deadline_penalty +
        0.25 * latency_score +
        0.3 * energy_score +
        0.2 * ce_score +
        0.1 * wait_score +
        0.15 * work_penalty
    )

    # Final sanitization: ensure finite, deterministic, shape-(N,)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    return score
