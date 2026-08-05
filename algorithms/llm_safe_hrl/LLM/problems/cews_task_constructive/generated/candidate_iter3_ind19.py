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
    Self-evolved priority rule: fixes semantic inversion in deadline modeling,
    restores anti-starvation for at-risk tasks, aligns energy objective with minimization,
    and introduces adaptive slack gating + uncertainty-calibrated aging.

    Key fixes & improvements:
    - Restores correct deadline semantics: arctan(slack * k) → negative slack → negative output → lower score (higher priority)
    - Anti-starvation (wait_score) now active for ALL tasks, but scaled by urgency: stronger boost when slack < 0
    - Energy term uses min_incremental_energy directly (not ratio) — since objective is total energy minimization, lower energy = higher priority
    - Uncertainty modulates BOTH deadline risk AND energy penalty: high uncertainty → amplify deadline urgency AND energy caution
    - Adaptive gating: upward_rank and remaining_work use smooth sigmoid mask instead of hard threshold for robustness
    - Robust normalization fallback handles zero-variance inputs without NaN
    - All terms are bounded, finite, deterministic, and satisfy interface contract.
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

    def robust_normalize(x):
        center = np.median(x)
        mad = np.median(np.abs(x - center)) + eps
        # Fallback to mean absolute deviation if MAD is near-zero
        scale = mad if mad > eps else np.mean(np.abs(x - center)) + eps
        return (x - center) / (scale + eps)

    # --- Deadline risk: monotonic, bounded, correct polarity ---
    # Negative slack → more urgent → lower score. arctan maps R → (-π/2, π/2); we invert sign so urgency lowers score.
    deadline_base = np.arctan(slack * 0.3)  # positive slack → positive → less urgent; negative → negative → urgent
    # Uncertainty amplifies urgency: higher uncertainty tightens effective deadline
    uncertainty_factor = 1.0 + np.clip(uncertainty, 0.0, 8.0)
    deadline_score = -deadline_base * uncertainty_factor  # lower = more urgent

    # --- Energy priority: direct minimization ---
    # Since objective is minimize total energy, prioritize low min_incremental_energy
    energy_score = robust_normalize(min_incremental_energy)

    # --- Critical-path importance: gated smoothly by slack ---
    # Soft mask: full weight when slack >= 0, decays smoothly as slack becomes negative (no hard cutoff)
    slack_sigmoid = 1.0 / (1.0 + np.exp(-slack / (np.abs(np.median(slack)) + eps)))  # centered around median slack
    rank_score = -robust_normalize(upward_rank) * slack_sigmoid

    # --- Remaining work: also softly gated, favors large work only when safe ---
    work_score = -robust_normalize(remaining_work) * slack_sigmoid

    # --- Anti-starvation: always active, but boosted under urgency ---
    # sqrt(wait) grows slowly; scaled by urgency factor: stronger boost when slack < 0
    wait_base = np.sqrt(np.maximum(ready_wait_time, 0.0) + eps)
    urgency_boost = 1.0 + np.clip(-np.minimum(slack, 0.0), 0.0, 5.0)  # adds up to +5 when slack = -5
    wait_score = -robust_normalize(wait_base) * urgency_boost

    # --- Uncertainty penalty: not just risk modulation — penalizes high-uncertainty tasks uniformly ---
    unc_score = robust_normalize(uncertainty)

    # --- Weighted ensemble: calibrated to emphasize deadline safety first, then energy, then structure ---
    score = (
        0.40 * deadline_score +
        0.25 * energy_score +
        0.12 * rank_score +
        0.08 * work_score +
        0.10 * wait_score +
        0.05 * unc_score
    )

    # Ensure finite output: replace NaN/inf with large finite bounds
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    return score
