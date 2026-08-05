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
    Self-evolved priority rule: deadline-feasibility preserving energy efficiency with calibrated risk gating and starvation fairness.
    
    Key improvements over v1:
    - Replaces aggressive risk_magnitude with *slack-scaled urgency*: (max(0, -slack) / (|median_slack| + eps)) × (1 + uncertainty),
      ensuring urgency grows sublinearly with lateness and avoids overshooting when slack is deeply negative.
    - Drops uncertainty-weighted energy-efficiency density (harmful noise amplifier); reverts to clean energy_per_sec = min_incremental_energy / (time_sum + eps),
      then applies *risk-aware scaling*: multiply by (1 + min(uncertainty, 0.5)) only for tasks with non-negative slack — avoids punishing uncertain-but-critical tasks.
    - Uses *linear starvation relief* (not sigmoid): normalized wait_ratio = ready_wait_time / (np.max(ready_wait_time) + eps), capped at 1.0,
      then negated and scaled — preserves monotonic fairness without gradient distortion.
    - Introduces *critical path dominance*: upward_rank × sqrt(remaining_work / (max_rw + eps)), emphasizing high-leverage, high-work nodes more robustly.
    - Adds *slack-tightness modulation*: all components are multiplied by a gate = 1.0 / (1.0 + exp(-2.0 * (median_abs_slack - abs_slack)/ (median_abs_slack + eps))),
      amplifying priority adjustments precisely when slack is tight (i.e., near deadline), not globally.
    - All normalizations use percentile-based robust_normalize; all divisions/ops eps-protected; nan/inf guarded.
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
        q75, q25 = np.percentile(x, 75), np.percentile(x, 25)
        iqr = q75 - q25 + eps
        center = np.median(x)
        return (x - center) / iqr
    
    # Slack-tightness gate: peaks when |slack| is small (tight deadlines)
    abs_slack = np.abs(slack)
    median_abs_slack = np.median(abs_slack) + eps
    slack_tightness_gate = 1.0 / (1.0 + np.exp(-2.0 * (median_abs_slack - abs_slack) / median_abs_slack))
    
    # Risk-adjusted urgency: sublinear, bounded, only active for late tasks
    late_mask = (slack < 0).astype(float)
    slack_penalty = np.maximum(0.0, -slack)  # non-negative penalty
    normalized_late_penalty = slack_penalty / (median_abs_slack + eps)
    urgency = normalized_late_penalty * (1.0 + np.clip(uncertainty, 0.0, 0.5)) * late_mask
    norm_urgency = -robust_normalize(urgency)  # lower score = higher priority
    
    # Clean energy efficiency: energy per second, risk-modulated only for safe tasks
    time_sum = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / time_sum
    # Apply uncertainty penalty only when slack >= 0 (avoid penalizing critical tasks)
    safe_mask = (slack >= 0).astype(float)
    risk_modulated_energy = energy_per_sec * (1.0 + np.clip(uncertainty, 0.0, 0.3) * safe_mask)
    norm_energy = -robust_normalize(risk_modulated_energy)
    
    # Critical path dominance: upward_rank * sqrt(remaining_work / max_rw)
    max_rw = np.max(remaining_work) + eps
    normalized_rw = remaining_work / max_rw
    critical_dominance = upward_rank * np.sqrt(normalized_rw + eps)
    norm_critical = -robust_normalize(critical_dominance)
    
    # Linear starvation relief: fair, monotonic, capped
    max_wait = np.max(ready_wait_time) + eps
    wait_ratio = np.clip(ready_wait_time / max_wait, 0.0, 1.0)
    norm_starvation = -robust_normalize(wait_ratio)
    
    # Communication risk penalty: only under tight slack and nonzero comm
    comm_risk_penalty = min_comm_time * uncertainty * (min_comm_time > eps).astype(float) * (abs_slack < median_abs_slack).astype(float)
    norm_comm_risk = robust_normalize(comm_risk_penalty)
    
    # Execution time baseline (lower exec → higher priority)
    norm_exec = robust_normalize(min_exec_time)
    
    # Assemble final score with slack-tightness gating applied to all terms
    score = (
        0.40 * norm_urgency +
        0.25 * norm_energy +
        0.15 * norm_critical +
        0.08 * norm_starvation +
        0.07 * norm_comm_risk +
        0.05 * norm_exec
    )
    score = score * slack_tightness_gate  # modulate entire priority signal by deadline pressure
    
    # Ensure finite output
    return np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
