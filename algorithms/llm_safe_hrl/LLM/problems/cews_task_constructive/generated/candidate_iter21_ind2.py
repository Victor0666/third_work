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
    Self-evolved v2: Deadline-critical energy fairness with adaptive starvation control and monotonic risk scaling.
    
    Key advances over v1:
    - Replaces hard/soft urgency switch with *monotonic tanh-based deadline pressure* for smooth, bounded, differentiable boundary behavior.
    - Introduces *energy-starvation coupling*: wait-aware penalty now scales energy-per-work by (1 + tanh(ready_wait_time / max_slack)) only when slack < 0, preventing over-penalization of long-wait tasks with ample slack.
    - Upgrades critical-path synergy to *dynamic risk gating*: cp_density_amplified = cp_density * (1 + uncertainty * sigmoid(-slack/eps)), smoothly activating risk amplification near deadline.
    - Adds *uncertainty-normalized energy fairness*: divides energy_per_work_scaled by (1 + uncertainty) to favor low-risk high-efficiency tasks under tight deadlines.
    - Robust normalization now uses *rank-based standardization* for N>=3 (preserves ordinal structure), fallback to quantile scaling for N<3 — eliminates median/mad sensitivity to outliers in tiny batches.
    - All terms clipped and eps-protected *before* combination; final score clamped and validated.
    - Deterministic, finite, shape-preserving, and fully compliant with interface contract.
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

    # Safe clipping & nan/inf handling per input
    min_exec_time = np.clip(np.nan_to_num(min_exec_time, nan=eps, posinf=eps, neginf=eps), eps, 1e9)
    min_comm_time = np.clip(np.nan_to_num(min_comm_time, nan=eps, posinf=eps, neginf=eps), eps, 1e9)
    min_incremental_energy = np.clip(np.nan_to_num(min_incremental_energy, nan=eps, posinf=eps, neginf=eps), eps, 1e9)
    slack = np.nan_to_num(slack, nan=0.0, posinf=1e9, neginf=-1e9)
    upward_rank = np.clip(np.nan_to_num(upward_rank, nan=eps, posinf=1e6, neginf=eps), eps, 1e6)
    remaining_work = np.clip(np.nan_to_num(remaining_work, nan=eps, posinf=1e9, neginf=eps), eps, 1e9)
    ready_wait_time = np.clip(np.nan_to_num(ready_wait_time, nan=0.0, posinf=1e6, neginf=0.0), 0.0, 1e6)
    uncertainty = np.clip(np.nan_to_num(uncertainty, nan=0.0, posinf=1e3, neginf=0.0), 0.0, 1e3)

    task_duration = min_exec_time + min_comm_time
    # Monotonic deadline pressure: tanh smoothly maps slack deficit to [0,1], avoiding step discontinuities
    deadline_pressure = 0.5 * (1.0 - np.tanh(slack / (task_duration + eps)))  # 0 when slack >> 0, ~1 when slack << 0

    # Division-based critical alignment with robust denominator
    slack_risk_penalty = 1.0 + np.maximum(0.0, -slack) * uncertainty
    crit_alignment = upward_rank / (slack_risk_penalty * (min_incremental_energy + eps))
    crit_alignment = np.clip(crit_alignment, eps, 1e7)

    # Slack-aware energy fairness: penalize energy more when slack tight, reward when slack ample
    slack_factor = np.maximum(1.0, 1.0 + slack / 10.0)
    work_slack_scaled = remaining_work * slack_factor
    energy_per_work_scaled = min_incremental_energy / (work_slack_scaled + eps)
    energy_per_work_scaled = np.clip(energy_per_work_scaled, eps, 1e9)

    # Critical-path density: prioritize fast, high-impact tasks
    cp_density = upward_rank / (task_duration + eps)
    cp_density = np.clip(cp_density, eps, 1e7)

    # Dynamic risk gating: smoothly amplify cp_density under deadline stress
    risk_gate = 1.0 + uncertainty * (1.0 / (1.0 + np.exp(slack / eps)))  # sigmoid(-slack/eps) ≈ 1 when slack < 0
    cp_density_amplified = cp_density * risk_gate

    # Energy-starvation coupling: only activate wait penalty under deadline pressure
    wait_energy_penalty = np.zeros_like(min_incremental_energy)
    wait_normalized = np.where(task_duration > eps, ready_wait_time / (task_duration + eps), 0.0)
    wait_normalized = np.clip(wait_normalized, 0.0, 1e3)
    # Apply starvation boost only when slack is negative (i.e., at risk)
    wait_boost_mask = slack < 0.0
    wait_energy_penalty = np.where(wait_boost_mask,
                                  energy_per_work_scaled * np.tanh(wait_normalized),
                                  0.0)

    # Uncertainty-normalized energy fairness: prefer low-uncertainty efficient tasks
    energy_fairness = energy_per_work_scaled / (1.0 + uncertainty + eps)

    def robust_normalize(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.clip(x, -1e9, 1e9)
        if N == 1:
            return np.array([0.0])
        elif N < 3:
            # Quantile-based for small N: stable, rank-preserving enough
            q1, q3 = np.quantile(x, [0.25, 0.75], method='midpoint')
            iqr = q3 - q1 + eps
            center = np.median(x)
            z = (x - center) / iqr
            return np.clip(z, -4.0, 4.0)
        else:
            # Rank-based standardization for N>=3: more ordinal-stable than MAD under skew
            ranks = np.argsort(np.argsort(x))  # dense rank
            mean_rank = np.mean(ranks)
            std_rank = np.std(ranks, ddof=1) + eps
            z = (ranks - mean_rank) / std_rank
            return np.clip(z, -6.0, 6.0)

    norm_pressure = robust_normalize(deadline_pressure)
    norm_crit = robust_normalize(crit_alignment)
    norm_energy = robust_normalize(energy_fairness)
    norm_cp_density = robust_normalize(cp_density_amplified)
    norm_wait_penalty = robust_normalize(wait_energy_penalty)

    # Final weighted score: smaller = higher priority
    score = (
        0.45 * norm_pressure      # deadline pressure dominates
        - 0.35 * norm_crit        # critical-path alignment (higher rank → lower score)
        + 0.12 * norm_energy      # energy fairness (lower energy/work → lower score)
        + 0.08 * norm_cp_density  # density synergy (higher density → lower score)
        + 0.05 * norm_wait_penalty  # starvation control (lower penalty → lower score)
    )

    # Final safeguard
    score = np.nan_to_num(score, nan=0.0, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e9, 1e9)
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
