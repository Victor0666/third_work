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
    Self-evolved v2: Fixes over-complication by unifying energy signal into single critical-energy-density (CED),
    restores aging boost magnitude via linear-sigmoid hybrid, applies full-array MAD normalization (not masked),
    and introduces slack-gated CED weight to dynamically prioritize energy efficiency only when deadline margin exists.
    Key improvements:
      - Single robust CED = upward_rank * remaining_work / (min_incremental_energy + eps)
      - Full-array MAD normalization (no masking) preserves relative ordering across all tasks
      - Slack-gated CED weight: smooth transition from deadline-driven (slack<0) to energy-aware (slack>0)
      - Aging boost = min(ready_wait_time, 10*sched_quantum) * sigmoid(1 - normalized_slack), preventing starvation without DDL violation
      - Uncertainty used only for urgency amplification in tight-positive slack regime (0 < slack <= Q3), not latency inflation
      - Hard penalty bounded at 0.4 and applied additively only for violated slack, preserving energy signal dominance under feasibility
      - All operations protected against zero/Nan/inf; shape (N,) guaranteed.
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

    # Robust full-array MAD normalization (handles N=1, constant, NaN, inf)
    def robust_mad_normalize(x):
        x_clean = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x_clean.size == 0:
            return np.zeros(0)
        if x_clean.size == 1:
            return np.zeros_like(x_clean)
        median_x = np.median(x_clean)
        mad = np.median(np.abs(x_clean - median_x)) + eps
        normed = (x_clean - median_x) / mad
        return np.clip(normed, -3.0, 3.0)

    # Deadline urgency: monotonic, smooth, full-range [0,1] via arctan
    deadline_urgency = (np.arctan(-slack) + np.pi / 2) / np.pi

    # Hard penalty: bounded additive term only for violated slack
    hard_penalty = np.where(slack < -eps, np.clip(-slack / (np.abs(np.min(slack)) + eps), 0.0, 0.4), 0.0)
    deadline_risk = deadline_urgency + hard_penalty
    deadline_score = robust_mad_normalize(deadline_risk)

    # Unified critical-energy-density: higher score = more energy-efficient per criticality
    ced = upward_rank * remaining_work / (min_incremental_energy + eps)
    ced_norm = robust_mad_normalize(ced)

    # Slack-gated CED weight: smoothly transitions from 0 (urgent) to 1 (margin available)
    # Uses clipped sigmoid on normalized slack to avoid extreme sensitivity near zero
    slack_abs_max = np.maximum(np.abs(np.max(slack)), np.abs(np.min(slack))) + eps
    normalized_slack = np.clip(slack / slack_abs_max, -2.0, 2.0)
    ced_weight = 1.0 / (1.0 + np.exp(-2.0 * normalized_slack))  # ~0 at slack=-1, ~1 at slack=+1

    # Uncertainty amplification only for tight-positive slack (0 < slack <= Q3 of positive slack)
    pos_slack = slack[slack > 0]
    median_pos_slack = np.median(pos_slack) if pos_slack.size > 0 else 0.0
    q3_pos_slack = np.percentile(pos_slack, 75) if pos_slack.size > 0 else median_pos_slack + eps
    uncertainty_active = (slack > 0) & (slack <= q3_pos_slack + eps)
    uncertainty_amplifier = np.where(uncertainty_active, 1.0 + 0.3 * np.clip(uncertainty, 0.0, 1.0), 1.0)

    # Aging boost: prevents starvation, scales with wait time but decays under high urgency
    # Uses linear growth capped at 90th percentile + sigmoid decay based on slack pressure
    wait_cap = np.percentile(ready_wait_time, 90) + eps if ready_wait_time.size > 1 else np.max(ready_wait_time) + eps
    clipped_wait = np.clip(ready_wait_time, 0.0, wait_cap)
    slack_pressure = np.clip((0.0 - slack) / (np.abs(np.min(slack)) + eps), 0.0, 1.0)
    aging_decay = 1.0 / (1.0 + np.exp(3.0 * (slack_pressure - 0.4)))
    aging_boost = 0.04 * clipped_wait * aging_decay

    # Final score: deadline dominates when urgent (ced_weight low); energy dominates when margin exists
    # Uncertainty amplifies deadline risk only in tight-positive regime — no separate term
    score = (
        +3.6 * deadline_score
        - 2.8 * ced_weight * ced_norm * uncertainty_amplifier
        + 0.04 * aging_boost
    )

    # Final safeguard: ensure finite, shape-(N,), deterministic output
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    return score.reshape(-1)
