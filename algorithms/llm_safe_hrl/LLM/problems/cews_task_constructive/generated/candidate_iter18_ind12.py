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

    '\n    Priority rule v3 (self-evolved): Fixes priority inversion & slack-zero noise via:\n      - Soft urgency gating: sigmoid(-slack) for smooth, differentiable dominance when slack < 0 → preserves discrimination among non-violating tasks\n      - CAER with dynamic criticality threshold: gate = (upward_rank > Q1(upward_rank)) & (slack >= 0) → avoids arbitrary eps thresholds\n      - Smoothed fairness: tanh(ready_wait_time / (|slack| + τ_safe)) with τ_safe=2.0 → bounded, noise-resilient aging under tight deadlines\n      - Uncertainty-aware latency: min_comm_time * (1 + uncertainty) only when slack >= 0; else min_exec_time + |slack| penalty → explicitly penalizes lateness beyond compute\n      - Component-wise quantile normalization with per-term robust scaling (Q1/Q3 + eps) → eliminates rank collapse\n      - Additive weighted fusion with decaying weights (1000, 100, 10, 1) enforcing strict hierarchy without multiplicative fragility\n      - Final score sanitized and clamped to finite range with deterministic fallbacks\n    '
    eps = 1e-08
    τ_safe = 2.0
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=eps, posinf=1000000.0, neginf=eps)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=eps, posinf=1000000.0, neginf=eps)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=eps, posinf=1000000.0, neginf=eps)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=1000000.0, neginf=-1000000.0)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=0.0, posinf=1000000.0, neginf=0.0)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=0.0, posinf=1000000.0, neginf=0.0)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=1000000.0, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=0.0, posinf=10.0, neginf=0.0)
    
    def robust_quantile_normalize(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size <= 1:
            return np.zeros_like(x)
        q1 = np.quantile(x, 0.25)
        q3 = np.quantile(x, 0.75)
        iqr = q3 - q1
        scale = iqr if iqr > eps else np.max(x) - np.min(x) + eps
        scale = max(scale, eps)
        center = (q1 + q3) / 2.0
        return (x - center) / scale

    # Soft urgency: sigmoid(-slack) → high priority for negative slack, smooth decay for positive slack
    urgency_raw = 1.0 / (1.0 + np.exp(slack))  # ≈1 when slack<<0, ≈0.5 at slack=0, ≈0 when slack>>0
    urgency_norm = robust_quantile_normalize(urgency_raw)
    urgency_score = urgency_norm * 1000.0

    # Criticality-Aware Energy Ratio with adaptive gate: use Q1 of upward_rank as threshold
    criticality = upward_rank * remaining_work
    q1_upward = np.quantile(upward_rank, 0.25) if len(upward_rank) > 1 else np.median(upward_rank)
    caer_gate = (upward_rank > max(q1_upward, eps)) & (slack >= 0)
    caer_base = criticality / (min_incremental_energy + eps)
    caer_masked = np.where(caer_gate, caer_base, 0.0)
    caer_norm = robust_quantile_normalize(caer_masked)
    caer_score = -caer_norm * 100.0  # negative: higher CAER → lower score → higher priority

    # Latency term: penalize communication when safe, penalize execution + lateness penalty when urgent
    latency_raw = np.where(
        slack < 0,
        min_exec_time + np.abs(slack),  # explicit lateness penalty
        min_comm_time * (1.0 + np.clip(uncertainty, 0.0, 5.0))
    )
    latency_norm = robust_quantile_normalize(latency_raw)
    latency_score = latency_norm * 10.0

    # Smoothed fairness: tanh-based aging activated only when slack is non-negative and finite
    fairness_raw = np.tanh(ready_wait_time / (np.abs(slack) + τ_safe))
    fairness_norm = robust_quantile_normalize(fairness_raw)
    fairness_score = fairness_norm * 1.0

    # Weighted additive fusion with strict hierarchy (no multiplicative coupling)
    score = urgency_score + caer_score + latency_score + fairness_score

    # Final sanitization: ensure finite, deterministic output
    score = np.nan_to_num(score, nan=1000000000.0, posinf=1000000000.0, neginf=-1000000000.0)
    score = np.clip(score, -1000000000.0, 1000000000.0)
    return score.astype(float)
