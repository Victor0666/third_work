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
    v3: Robust lexicographic safety-first priority with:
      - Hard violation override (slack < 0) → absolute minimum score (not just large negative)
      - Triple-gated energy-awareness: CED active *only* when robust_slack > median_robust_pos AND uncertainty < 0.5
      - Latency-criticality coupling: latency_cost weighted by upward_rank * tanh(slack / (median_slack_pos + eps))
      - Fairness redefined as *relative aging*: ready_wait_time / (max(1.0, median_slack_pos, -slack)) → prevents explosion near deadline
      - Uncertainty amplification now adaptive: factor = 1 + 0.8 * clipped_uncert * sigmoid(-robust_slack)
      - All normalizations use MAD with strict finite bounds, N=1 guard, and double-clipping to [-2.8, 2.8]
      - Final score strictly monotonic in slack for feasible region; deterministic, finite, and numerically safe.
    """
    eps = 1e-09
    def sanitize(x):
        x = np.asarray(x, dtype=float)
        return np.nan_to_num(x, nan=eps, posinf=1e9, neginf=1e-9)

    min_exec_time = sanitize(min_exec_time)
    min_comm_time = sanitize(min_comm_time)
    min_incremental_energy = sanitize(min_incremental_energy)
    slack = sanitize(slack)
    upward_rank = sanitize(upward_rank)
    remaining_work = sanitize(remaining_work)
    ready_wait_time = sanitize(ready_wait_time)
    uncertainty = sanitize(uncertainty)

    # Robust slack: slack - 2*uncertainty, clipped to avoid overflow
    robust_slack = np.clip(slack - 2.0 * uncertainty, -1e8, 1e8)

    # Hard violation: slack < 0 → assign absolute lowest possible score
    violation_mask = slack < 0
    base_score = np.full_like(slack, 0.0)

    # Compute median_slack_pos for gating (positive slack only); fallback if none exist
    positive_slack = slack[slack > 0]
    median_slack_pos = np.median(positive_slack) if positive_slack.size > 0 else 1.0
    median_slack_pos = max(median_slack_pos, eps)

    # Urgency: smooth, bounded, and slack-monotonic using tanh-based inverse margin
    # Ensures urgency → 1 as slack → -∞, and → 0 as slack → +∞; centered at median_slack_pos
    margin = (median_slack_pos - slack) / (median_slack_pos + eps)
    urgency_raw = 0.5 * (1.0 - np.tanh(margin))
    urgency_raw = np.clip(urgency_raw, 0.0, 1.0)

    # Adaptive uncertainty amplification: stronger penalty when robust_slack is negative or small positive
    # sigmoid(-robust_slack) ≈ 1 when robust_slack << 0, ≈ 0.5 at 0, decays as robust_slack increases
    sig_factor = 1.0 / (1.0 + np.exp(robust_slack))
    clipped_uncert = np.clip(uncertainty, 0.0, 1.0)
    amplification_factor = 1.0 + 0.8 * clipped_uncert * sig_factor
    urgency = urgency_raw * amplification_factor

    # Critical Energy Density (CED): (upward_rank * remaining_work) / (energy * latency)
    # Gated by *both* robust feasibility AND low uncertainty (confidence filter)
    latency_cost = np.clip(min_exec_time + min_comm_time + eps, eps, 1e6)
    ced_numerator = upward_rank * remaining_work + eps
    ced_denominator = (min_incremental_energy + eps) * latency_cost
    ced_base = ced_numerator / ced_denominator

    ced_gate = ((robust_slack > median_slack_pos) & (uncertainty < 0.5)).astype(float)
    ced_masked = ced_base * ced_gate

    # CP boost: upward_rank * normalized_work * exp(-uncertainty) * time-sensitivity decay
    # Decay factor uses robust_slack to avoid over-prioritizing deep-negative tasks
    cp_decay = 1.0 / (1.0 + np.abs(robust_slack) + eps)
    cp_boost_raw = upward_rank * (remaining_work / (np.median(remaining_work) + eps)) * np.exp(-uncertainty) * cp_decay * ced_gate
    cp_masked = cp_boost_raw

    # Fairness: relative aging — wait time normalized against *available safety margin*
    # Uses max(1.0, median_slack_pos, -slack) to prevent division by zero *and* avoid artificial inflation when slack is negative
    fairness_denom = np.maximum(1.0, np.maximum(median_slack_pos, -slack + eps))
    fairness_raw = ready_wait_time / (fairness_denom + eps)
    # Apply soft saturation to avoid unbounded growth under large wait times
    fairness = np.tanh(0.7 * fairness_raw)

    # Normalize all components safely with MAD, strict clipping, and N=1 handling
    def safe_mad_normalize(x):
        x_clipped = np.clip(x, -1e6, 1e6)
        if x_clipped.size == 1:
            return np.array([0.0])
        med = np.median(x_clipped)
        mad = np.median(np.abs(x_clipped - med)) + eps
        normed = (x_clipped - med) / mad
        return np.clip(normed, -2.8, 2.8)

    urgency_norm = safe_mad_normalize(urgency)
    ced_norm = safe_mad_normalize(ced_masked)
    cp_norm = safe_mad_normalize(cp_masked)
    fairness_norm = safe_mad_normalize(fairness)

    # Weighted combination: urgency dominates; CED and CP are energy-saving bonuses (negative weight); fairness minor correction
    score = (
        +3.2 * urgency_norm
        - 1.3 * ced_norm
        - 0.6 * cp_norm
        + 0.04 * fairness_norm
    )

    # Apply hard violation override: set score to absolute minimum (not just large negative)
    # Ensures np.argmin will *always* pick violated tasks first
    score = np.where(violation_mask, -1e12, score)

    # Final sanitization: guarantee finite, deterministic output
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    score = np.clip(score, -1e12, 1e12)

    return score
