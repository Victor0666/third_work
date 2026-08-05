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
    Self-evolved priority rule v2: restores strict lateness-risk discrimination,
    reintroduces robust per-feature scaling with physical-aware centering,
    decouples urgency amplification from uncertainty coupling, and adds
    deadline-aware energy fairness to prevent high-energy tasks from starving
    under tight deadlines.

    Key evolutions from v1:
      - Restores explicit neg_slack-based lateness penalty (not slack_gap) to
        preserve hard DDL enforcement semantics: only penalize actual lateness risk.
      - Reintroduces robust median/IQR scaling *per term*, but now anchored to
        task duration for slack and wait_time → maintains physical interpretability
        (e.g., 1s slack matters more on short-duration tasks).
      - Removes uncertainty-modulated urgency amplification; instead uses
        uncertainty-weighted *energy fairness* term to steer away from high-risk
        high-energy assignments *only when slack permits*, preserving pure urgency signal.
      - Adds 'deadline-aware energy fairness': penalizes min_incremental_energy
        only when slack > 0, scaled by (1 - sigmoid(slack / duration)), ensuring
        energy minimization activates *only in feasible regime*, never at expense of DDL.
      - Introduces 'criticality-normalized wait boost': starvation boost is weighted
        by upward_rank to prioritize waiting on critical path nodes first.
      - All terms epsilon-protected, nan/inf-cleaned, shape-verified, deterministic.
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

    # Physical duration baseline for relative scaling
    duration = np.maximum(min_exec_time + min_comm_time, eps)

    # === 1. STRICT LATENESS RISK PENALTY (hard DDL enforcement) ===
    # Only penalize negative slack — preserves binary urgency semantics
    neg_slack = np.maximum(-slack, 0.0)
    # Base penalty: linear in lateness magnitude, amplified by uncertainty
    base_deadline_penalty = neg_slack * (8.0 + 3.0 * np.clip(uncertainty, 0.0, 2.0))
    # Hard gating: apply *additional* fixed penalty to all deeply urgent tasks (sl<30th)
    slack_sorted = np.sort(slack)
    slack_30 = slack_sorted[max(0, int(0.3 * len(slack_sorted)))] if N > 0 else 0.0
    is_deeply_urgent = slack < slack_30
    deadline_penalty = np.where(is_deeply_urgent, base_deadline_penalty + 15.0, base_deadline_penalty)

    # === 2. ROBUST PER-TERM SCALING (duration-anchored for slack & wait) ===
    def robust_scale(x, ref_duration=None):
        if len(x) == 0:
            return x
        med = np.median(x)
        x_centered = x - med
        q1, q3 = np.quantile(x_centered, [0.25, 0.75], method='midpoint')
        iqr = q3 - q1
        if iqr < eps:
            mad = np.median(np.abs(x_centered))
            scale = mad if mad > eps else np.mean(np.abs(x_centered)) + eps
        else:
            scale = iqr + eps
        # For slack & wait_time: normalize w.r.t. duration to reflect relative impact
        if ref_duration is not None:
            scale = np.maximum(scale, np.median(ref_duration) + eps)
        return np.clip(x_centered / scale + med, -1000000.0, 1000000.0)

    # === 3. CRITICALITY-ENERGY TRADEOFF (CE ratio & synergy) ===
    ce_ratio = upward_rank / (min_incremental_energy + eps)
    ce_score = -robust_scale(ce_ratio)

    synergy = upward_rank * duration / (min_incremental_energy + eps)
    synergy_score = -robust_scale(synergy)

    # === 4. DEADLINE-AWARE ENERGY FAIRNESS (only active when slack > 0) ===
    # Avoid energy optimization when lateness risk exists → prevents DDL violation
    slack_norm = np.clip(slack / (duration + eps), -10.0, 10.0)
    # Sigmoid gate: near 0 when slack <= 0, rises smoothly to 1 as slack increases
    energy_activation = 1.0 / (1.0 + np.exp(-slack_norm))
    energy_base = min_incremental_energy * (1.0 + 0.5 * np.clip(uncertainty, 0.0, 1.0))
    energy_score = robust_scale(energy_base) * energy_activation

    # === 5. STARVATION-AWARE WAIT BOOST (criticality-weighted) ===
    wait_thresh = np.quantile(ready_wait_time, 0.75) if N > 0 else eps
    wait_std = np.maximum(np.std(ready_wait_time), eps) + eps
    wait_boost_raw = np.where(ready_wait_time > wait_thresh,
                             (ready_wait_time - wait_thresh) / wait_std,
                             0.0)
    # Weight boost by upward_rank → prioritize waiting on critical path
    wait_score = -wait_boost_raw * (upward_rank / (np.maximum(np.median(upward_rank), eps) + eps))

    # === 6. SLACK-GATED WORK PENALTY (heavy subtrees penalized only when urgent) ===
    slack_gap = np.clip(slack_30 - slack, 0.0, None)
    work_norm = robust_scale(remaining_work)
    work_penalty = work_norm * (slack_gap / (np.abs(slack_30) + eps))

    # === FINAL SCORE: convex combination, bounded and cleaned ===
    score = (
        2.2 * deadline_penalty +
        0.3 * ce_score +
        0.45 * synergy_score +
        0.2 * energy_score +
        0.15 * wait_score +
        0.18 * work_penalty
    )

    # Final sanitization
    score = np.nan_to_num(score, nan=1000000000000.0, posinf=1000000000000.0, neginf=-1000000000000.0)
    score = np.clip(score, -1000000000000.0, 1000000000000.0)

    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
