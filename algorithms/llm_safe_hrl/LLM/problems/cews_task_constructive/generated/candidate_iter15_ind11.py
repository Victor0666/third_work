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
    v2: Refined deadline-hard priority with calibrated uncertainty integration,
         adaptive energy gating, median-robust fairness, and critical-path-aware risk.

    Key self-evolution improvements:
    - Replaces over-penalizing 'robust_slack = slack - 2*uncertainty' with
      'calibrated_slack = slack - uncertainty * clip(uncertainty, 0, 0.5)' →
      avoids excessive degradation for high-uncertainty/low-risk tasks.
    - CED gating uses (slack >= -uncertainty) instead of robust_slack >= 0 →
      allows early energy optimization when uncertainty is bounded and slack is near-zero.
    - Risk penalty now weighted by upward_rank to prioritize mitigation on critical paths.
    - Fairness scaling uses percentile-based (90th) negative slack instead of min →
      suppresses noise from outliers and strengthens starvation relief only under broad pressure.
    - Adds urgency-preserving slack normalization: maps raw slack into [-1, 1] via tanh(slack/5),
      then shifts & scales to emphasize urgency asymmetry (higher penalty for violation).
    - All terms use safe_mad_normalize with strict N=1 handling and finite-range clamping.
    - Final weights tuned for hierarchy: deadline (5.6) >> energy (3.1) >> critical-path (1.45) >> risk (0.65) >> fairness (0.12)
    """
    eps = 1e-08
    def clean_array(x):
        x = np.asarray(x, dtype=float)
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    min_exec_time = clean_array(min_exec_time)
    min_comm_time = clean_array(min_comm_time)
    min_incremental_energy = clean_array(min_incremental_energy)
    slack = clean_array(slack)
    upward_rank = clean_array(upward_rank)
    remaining_work = clean_array(remaining_work)
    ready_wait_time = clean_array(ready_wait_time)
    uncertainty = clean_array(uncertainty)

    # Calibrated slack: uncertainty scaled only up to 0.5 to avoid over-penalization
    capped_uncertainty = np.clip(uncertainty, 0.0, 0.5)
    calibrated_slack = slack - uncertainty * capped_uncertainty
    calibrated_slack = np.clip(calibrated_slack, -1000000000000.0, 1000000000000.0)

    # Urgency mapping: tanh-based symmetric compression + asymmetric shift for violation emphasis
    # Maps slack ∈ ℝ → [-1, 1], then shifts negative region downward: violation = -1.2 to -1.0, safety = -0.2 to +1.0
    urgency_base = np.tanh(calibrated_slack / 5.0)
    urgency_shifted = np.where(
        calibrated_slack < 0,
        urgency_base - 0.2,
        urgency_base + 0.2
    )
    # Clamp to enforce finite range and prevent overflow
    urgency_score = np.clip(urgency_shifted, -1.25, 1.25)

    def safe_mad_normalize(x):
        if x.size == 1:
            return np.zeros_like(x)
        x_clipped = np.clip(x, -1000000.0, 1000000.0)
        med = np.median(x_clipped)
        mad = np.median(np.abs(x_clipped - med)) + eps
        if mad < eps:
            return np.zeros_like(x)
        normed = (x_clipped - med) / mad
        return np.clip(normed, -4.0, 4.0)

    # Deadline urgency score — normalized, lower = safer, higher = more urgent
    deadline_score = safe_mad_normalize(urgency_score)

    # CED gating: allow energy optimization when slack >= -uncertainty (not just >= 0)
    # Ensures early efficiency without sacrificing feasibility margin
    ced_gate = ((slack >= -capped_uncertainty) & (upward_rank > eps)).astype(float)
    ced_numerator = upward_rank * remaining_work + eps
    ced_denominator = (min_incremental_energy + eps) * (min_exec_time + min_comm_time + eps)
    ced_raw = np.where(ced_gate > 0.0, ced_numerator / ced_denominator, 0.0)
    ced_norm = safe_mad_normalize(ced_raw)

    # Upward rank contribution: only active when calibrated_slack > 0 AND work > 0
    upward_rank_active = np.where((calibrated_slack > 0) & (remaining_work > eps), upward_rank, 0.0)
    upward_rank_norm = safe_mad_normalize(upward_rank_active)

    # Risk penalty: scaled by upward_rank to focus mitigation on critical paths
    # Only applied when calibrated_slack < 0 AND uncertainty > eps
    risk_raw = np.where(
        (calibrated_slack < 0) & (uncertainty > eps),
        (-calibrated_slack) * np.clip(uncertainty, 0.0, 0.7) * upward_rank,
        0.0
    )
    risk_penalty_norm = safe_mad_normalize(risk_raw)

    # Fairness: activated only under deadline pressure (calibrated_slack < 0)
    # Scaled by 90th percentile of negative calibrated_slack (more robust than min)
    neg_calibrated_slack = calibrated_slack[calibrated_slack < 0]
    if len(neg_calibrated_slack) == 0:
        fairness_scale_base = 1.0
    else:
        fairness_scale_base = np.percentile(neg_calibrated_slack, 90)  # most severe common pressure
    fairness_scale = np.clip(-fairness_scale_base, 0.0, 1000000.0) + eps
    wait_scale = np.clip(ready_wait_time / fairness_scale, 0.0, 20.0)
    fairness_boost = np.tanh(wait_scale)
    fairness_boost = np.clip(fairness_boost, 0.0, 0.15)
    fairness_norm = safe_mad_normalize(fairness_boost)

    # Final weighted combination: smaller score = higher priority
    w_deadline = 5.6
    w_ced = 3.1
    w_upward = 1.45
    w_risk = 0.65
    w_fairness = 0.12

    score = (
        w_deadline * deadline_score
        - w_ced * ced_norm
        - w_upward * upward_rank_norm
        + w_risk * risk_penalty_norm
        + w_fairness * fairness_norm
    )

    # Final sanitization: guarantee finiteness, determinism, and shape
    score = np.nan_to_num(score, nan=1000000000000.0, posinf=1000000000000.0, neginf=-1000000000000.0)
    score = np.clip(score, -1000000000000.0, 1000000000000.0)
    return score
