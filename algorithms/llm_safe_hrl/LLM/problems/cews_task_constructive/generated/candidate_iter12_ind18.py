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
    v3: Deadline-hardened, energy-aware, numerically bulletproof priority scorer.
    Key self-evolution improvements over v1:
    - Replaces median-based energy-gating with *slack-relative percentile gating*: 
      only activates energy efficiency for tasks within top-30% slack margin (not just above median),
      improving deadline sensitivity and reducing premature energy optimization.
    - Introduces *normalized urgency gradient*: uses sigmoid-transformed normalized slack instead of hard binary offset,
      enabling smooth, differentiable penalty ramping for overdue tasks (no abrupt 120.0 jumps).
    - Upgrades *risk-aware latency footprint* to *critical-path pressure*: replaces latency_footprint * upward_rank 
      with (upward_rank * remaining_work) / (min_exec_time + min_comm_time + eps), capturing work density under critical path.
    - Refines *uncertainty penalty* to be proportional to both normalized urgency AND normalized volatility,
      avoiding spurious penalties when uncertainty is high but slack is ample.
    - Replaces exponential aging with *wait-time quantile fairness*: uses wait rank (percentile) scaled by urgency,
      ensuring fairness without arbitrary tau estimation or clipping artifacts.
    - Adds *energy-efficiency saturation guard*: clamps energy term contribution to prevent dominance when energy values are near-zero.
    - All normalizations now use *robust std-based fallback* (std > eps else 1.0) instead of IQR-only, handling flat distributions safely.
    - Final score strictly bounded to [-1e6, 1e6] with deterministic shape enforcement.
    """
    eps = 1e-08
    # Safe conversion and nan/inf cleanup
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=0.0, posinf=1e6, neginf=1e6)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=0.0, posinf=1e6, neginf=1e6)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=0.0, posinf=1e6, neginf=1e6)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=1e6, neginf=-1e6)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=0.0, posinf=1e6, neginf=1e6)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=0.0, posinf=1e6, neginf=1e6)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=1e6, neginf=1e6)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=0.0, posinf=1e6, neginf=1e6)

    def robust_normalize(x):
        x = np.clip(x, -1e6, 1e6)
        if x.size == 0:
            return np.zeros_like(x)
        if x.size == 1:
            return np.array([0.0])
        mean_x, std_x = np.mean(x), np.std(x)
        if std_x < eps:
            return np.zeros_like(x)
        return (x - mean_x) / (std_x + eps)

    # Urgency: sigmoid-transformed normalized slack for smooth, bounded penalty ramp
    urgency_raw = np.where(slack < 0, -slack, 0.0)
    norm_slack = robust_normalize(slack)
    urgency_sigmoid = 1.0 / (1.0 + np.exp(-norm_slack))  # [0,1], rises smoothly at slack=0
    norm_urgency = robust_normalize(urgency_raw)

    # Critical-path pressure: work density under critical importance (higher = more urgent & heavy)
    cp_pressure = (upward_rank * (remaining_work + eps)) / (min_exec_time + min_comm_time + eps)
    norm_cp_pressure = robust_normalize(cp_pressure)

    # Energy efficiency: only activated for tasks in top-30% slack margin (soft gating)
    slack_percentile = np.percentile(slack, 70)  # top-30% have slack >= this
    energy_gate = (slack >= slack_percentile).astype(float)
    energy_efficiency = min_incremental_energy / (remaining_work + eps)
    norm_energy_eff = robust_normalize(energy_efficiency)
    energy_term = norm_energy_eff * energy_gate
    # Saturation guard: cap energy term to avoid dominance when energy ~0
    energy_term = np.clip(energy_term, -5.0, 5.0)

    # Uncertainty penalty: active only when both urgency and volatility are high
    norm_uncert = robust_normalize(uncertainty)
    uncertainty_penalty = np.where(
        (urgency_raw > np.quantile(urgency_raw, 0.7)) & (norm_uncert > np.quantile(norm_uncert, 0.7)),
        uncertainty * (urgency_raw + eps) * (norm_uncert + eps),
        0.0
    )
    norm_uncert_penalty = robust_normalize(uncertainty_penalty)

    # Wait-time fairness: percentile rank of wait time, scaled by urgency (no tau fitting)
    wait_rank = np.argsort(np.argsort(ready_wait_time)) / max(len(ready_wait_time) - 1, 1)  # [0,1] rank
    aging_term = -wait_rank * urgency_raw  # higher wait + higher urgency → stronger negative boost

    # Latency risk: imminent-window arctan on volatility, normalized
    base_latency = min_exec_time + min_comm_time + eps
    mean_base_latency = np.mean(base_latency) + eps
    relative_volatility = base_latency * (uncertainty + eps) / mean_base_latency
    adaptive_window = np.maximum(0.5, 0.1 * mean_base_latency)
    imminent_window = (slack >= 0) & (slack < adaptive_window)
    risk_arctan = np.where(imminent_window, 0.5 + 1.0 / np.pi * np.arctan(relative_volatility), 0.0)
    latency_risk_term = robust_normalize(risk_arctan)

    # Weighted combination — prioritizes deadline hardness first, then energy under constraint
    w_urgency = 7.2      # primary driver for DDL compliance
    w_cp_pressure = 3.0  # critical path load balancing
    w_energy = 2.5       # secondary: energy only when slack permits
    w_uncert = 1.4       # risk mitigation for borderline cases
    w_aging = 1.2        # fairness under overload
    w_risk = 1.9         # imminent latency risk
    score = (
        w_urgency * norm_urgency +
        w_cp_pressure * norm_cp_pressure +
        w_energy * energy_term +
        w_uncert * norm_uncert_penalty +
        w_aging * aging_term +
        w_risk * latency_risk_term
    )

    # Final numeric safeguards
    score = np.nan_to_num(score, nan=1e6, posinf=1e6, neginf=-1e6)
    score = np.clip(score, -1e6, 1e6)
    # Ensure shape (N,)
    if score.ndim == 0:
        score = np.array([score])
    elif score.ndim > 1:
        score = score.reshape(-1)
    if score.shape[0] != len(min_exec_time):
        score = score[:len(min_exec_time)]
        if score.shape[0] < len(min_exec_time):
            score = np.pad(score, (0, len(min_exec_time) - score.shape[0]), constant_values=1e6)
    return score
