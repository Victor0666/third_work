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

    eps = 1e-08
    N = len(slack)
    def clean_array(x):
        x = np.asarray(x, dtype=float)
        return np.nan_to_num(x, nan=eps, posinf=1000000.0, neginf=eps)
    min_exec_time = clean_array(min_exec_time)
    min_comm_time = clean_array(min_comm_time)
    min_incremental_energy = clean_array(min_incremental_energy)
    slack = clean_array(slack)
    upward_rank = clean_array(upward_rank)
    remaining_work = clean_array(remaining_work)
    ready_wait_time = clean_array(ready_wait_time)
    uncertainty = clean_array(uncertainty)
    
    # Restored robust quantile scaling with stable [-1.0, 1.0] bounds (v0-level stability)
    def quantile_scale(x):
        if x.size == 0:
            return np.zeros_like(x)
        if x.size == 1:
            return np.array([0.0])
        q1 = np.quantile(x, 0.25) + eps
        q3 = np.quantile(x, 0.75) + eps
        iqr = q3 - q1
        scaled = (x - (q1 + q3) / 2) / (iqr * 0.5 + eps)
        return np.clip(scaled, -1.0, 1.0)  # Tighter than v1, more monotonic than v0
    
    slack_abs = np.abs(slack)
    slack_med = np.median(slack_abs) + eps
    
    # Improved urgency: sharper tanh gradient near zero slack + bounded plateau for deep violation
    # Fixes v1's flattened response — restores sensitivity at critical boundary (slack ≈ 0)
    urgency_raw = np.where(
        slack <= 0,
        1.0 + 0.8 * np.tanh(np.clip(-slack / (slack_med + eps), 0.0, 3.0)),
        np.clip(1.0 - (slack_abs - slack_med) / (slack_med + eps), 0.0, 1.0)
    )
    norm_urgency = quantile_scale(urgency_raw)
    urgency_term = -16.5 * norm_urgency  # Moderated coefficient: balances signal vs noise
    
    exec_comm_sum = min_exec_time + min_comm_time + eps
    
    # Horizon weight: exponential only when slack ≤ 0, but with capped exponent (prevents overflow & over-suppression)
    horizon_weight = np.where(
        slack <= 0,
        np.exp(np.clip(-slack / (slack_med + eps), 0.0, 2.5)),  # Reduced max exponent from 4.0→2.5
        1.0
    )
    
    # Critical-path density: use *normalized* upward_rank to reduce skew from outlier magnitudes
    norm_upward_rank = (upward_rank - np.min(upward_rank) + eps) / (np.max(upward_rank) - np.min(upward_rank) + eps)
    cp_density_raw = (norm_upward_rank + eps) * (remaining_work + eps) * horizon_weight / exec_comm_sum
    
    # Uncertainty penalty only applied *outside safe zone*: slack > 0 AND uncertainty high → delays non-critical comms-heavy tasks
    unc_q90 = np.quantile(uncertainty, 0.9) + eps
    comm_unc_penalty = np.where(
        (slack > 0) & (uncertainty > unc_q90),
        min_comm_time * np.clip(uncertainty, 0.0, 4.0) * 0.15,  # Stronger penalty, narrower trigger
        0.0
    )
    cp_density_penalized = cp_density_raw + comm_unc_penalty / exec_comm_sum
    norm_cp_density = quantile_scale(cp_density_penalized)
    cp_density_term = 3.3 * norm_cp_density  # Slightly reduced weight for better energy balance
    
    # Energy-latency tradeoff: prefer ELTR unless variance is negligible → then fallback to LAED
    eltr_raw = (min_incremental_energy + eps) / (exec_comm_sum + eps)
    eltr_var = np.var(eltr_raw) if N > 1 else 0.0
    laed_raw = (min_incremental_energy + eps) / ((remaining_work + eps) * (exec_comm_sum + eps))
    use_laed = (eltr_var < 1e-09) | (np.mean(min_incremental_energy) < eps) | (np.mean(exec_comm_sum) < eps)
    energy_raw = np.where(use_laed, laed_raw, eltr_raw)
    
    # Energy gating: tighter slack-abs dependence — activates earlier (at 0.5×slack_med) and saturates faster
    energy_gate = np.clip(
        0.15 + 0.85 * (1.0 - np.tanh(np.clip(slack_abs / (slack_med + eps), 0.0, 5.0))),
        0.15, 1.0
    )
    energy_gated = energy_raw * energy_gate
    norm_energy = quantile_scale(energy_gated)
    energy_term = -2.6 * norm_energy  # Slightly less aggressive than v1 → preserves deadline priority
    
    # Fairness: wait-relative term now uses *dynamic threshold* (adaptive to workflow scale)
    wait_rel = ready_wait_time / (slack_abs + eps)
    wait_q75 = np.quantile(wait_rel, 0.75) + eps
    wait_threshold = np.maximum(wait_q75, 0.2)  # Lower floor than v1 (0.25→0.2) for responsiveness
    wait_saturation = np.clip(wait_rel / (wait_threshold + eps), 0.0, 1.0)
    wait_compressed = np.log1p(wait_saturation * 1.8)  # Softer compression than v1 (2.0→1.8)
    norm_wait = quantile_scale(wait_compressed)
    fairness_term = -0.45 * norm_wait  # Balanced weight: avoids starvation without over-prioritizing
    
    # Uncertainty gating: stricter dual condition + clipped impact → prevents dilution
    slack_q40 = np.quantile(slack, 0.4) + eps  # Earlier threshold (50→40%) for proactive risk handling
    unc_q60 = np.quantile(uncertainty, 0.6) + eps
    unc_boost = np.where(
        (slack < slack_q40) & (uncertainty > unc_q60),
        np.clip(uncertainty * 0.07, 0.0, 0.06),  # Tighter cap (0.07→0.06)
        0.0
    )
    norm_unc = quantile_scale(unc_boost)
    uncertainty_term = 0.06 * norm_unc  # Reduced weight for precision
    
    # Violation penalty: smooth, bounded, and *non-saturating* near extreme lateness
    # Uses linear-tanh hybrid to preserve ordering even for large negative slack
    violation_raw = np.clip(-slack, 0.0, 6.0 * slack_med)
    violation_penalty = (
        100000.0 * np.tanh(violation_raw / (slack_med + eps)) +
        20000.0 * np.clip(-slack / (slack_med + eps) - 3.0, 0.0, np.inf)
    )
    violation_term = violation_penalty
    
    score = urgency_term + cp_density_term + energy_term + fairness_term + uncertainty_term + violation_term
    score = np.nan_to_num(score, nan=100000000.0, posinf=100000000.0, neginf=-100000000.0)
    score = np.clip(score, -10000000.0, 10000000.0)
    return score.astype(float).reshape(-1)
