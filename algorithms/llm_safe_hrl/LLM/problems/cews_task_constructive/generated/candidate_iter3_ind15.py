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

    'Self-evolved priority rule: deadline-safe, risk-aware, and critical-path-precise.\n\n    Key self-evolution improvements:\n    - Replaces adaptive threshold with *slack-margin gating*: penalty only when slack < -0.1 sec,\n      ensuring hard DDL safety without premature penalization of marginally tight tasks.\n    - Introduces *slack-weighted upward_rank*: upward_rank * sigmoid(-slack/5.0) to preserve\n      high priority for critical nodes even when slack is small-but-positive (e.g., 0.5s left),\n      avoiding collapse of urgency near deadline.\n    - Energy term now uses *joules-per-critical-time*: min_incremental_energy / (min_exec_time + min_comm_time + eps),\n      decoupled from remaining_work to avoid biasing low-work critical tasks downward.\n    - Aging term is *slack-conditioned*: tanh(0.3 * ready_wait_time / (|slack| + 1.0)) —\n      accelerates aging only when slack is tight, preventing starvation *only when needed*.\n    - Risk amplification applied *only to deadline_risk*, not energy — avoids noise amplification\n      under low uncertainty while preserving sensitivity to deadline pressure under high uncertainty.\n    - All IQR normalizations use robust fallback (std or range) and explicit zero-IQR guard.\n    - Final score strictly bounded via clip() to prevent extreme outliers from dominating ranking.\n    '
    eps = 1e-08
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    
    def normalize_iqr(x):
        if x.size == 0:
            return np.zeros_like(x)
        if x.size == 1:
            return np.zeros_like(x, dtype=float)
        q75, q25 = np.percentile(x, [75, 25], method='midpoint')
        iqr = q75 - q25
        if iqr < eps:
            # Fallback: use std if IQR near-zero; if std also zero, use range
            std_val = np.std(x)
            scale = std_val if std_val > eps else (np.max(x) - np.min(x) + eps)
        else:
            scale = iqr
        center = np.median(x)
        return (x - center) / (scale + eps)
    
    # Hard deadline safety: penalty only when slack < -0.1s (hard violation threshold)
    deadline_risk = np.where(slack < -0.1, -slack, 0.0)
    
    # Slack-weighted criticality: preserves urgency for tasks near deadline, even if slack > 0
    slack_sigmoid = 1.0 / (1.0 + np.exp(-slack / 5.0))  # ~0.5 at slack=0, ~0.99 at slack=25
    weighted_upward_rank = upward_rank * slack_sigmoid
    
    # Energy efficiency: joules per actual execution+comm time (not diluted by descendant work)
    exec_comm_safe = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_time = min_incremental_energy / exec_comm_safe
    
    # Slack-conditioned aging: fairness only matters when slack is tight
    aging_denom = np.abs(slack) + 1.0  # prevents division by zero, smoothens
    aging_boost = np.tanh(0.3 * ready_wait_time / aging_denom)
    
    # Risk amplification only on deadline risk — avoids noise in energy term under low uncertainty
    risk_amp = 1.0 + 2.0 * (1.0 / (1.0 + np.exp(-uncertainty + 1.0)))
    
    # Normalize components
    norm_deadline_risk = normalize_iqr(deadline_risk)
    norm_weighted_rank = normalize_iqr(weighted_upward_rank)
    norm_energy = normalize_iqr(energy_per_time)
    norm_exec = normalize_iqr(min_exec_time)
    norm_comm = normalize_iqr(min_comm_time)
    norm_uncert = normalize_iqr(uncertainty)
    
    # Weighted linear combination: emphasizes deadline safety & critical path first
    score = (
        0.45 * norm_deadline_risk * risk_amp +
        0.25 * norm_weighted_rank +
        0.15 * norm_energy -
        0.08 * norm_exec -
        0.07 * norm_comm -
        0.05 * aging_boost +
        0.05 * norm_uncert
    )
    
    # Strictly bound output to prevent ranking collapse due to outliers
    score = np.clip(score, -1e12, 1e12)
    return np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
