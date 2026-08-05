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

    '''
    v2 priority rule: Urgency-first with slack-aware criticality gating + stabilized energy penalty +
    MAD-robust fairness *only when N>3* + uncertainty as modulator (not driver) + fallback to percentile norm.

    Key self-evolution improvements:
    - Replaces fragile MAD normalization with adaptive choice: use MAD only when N>3 AND variance > eps, else safe percentile-clipped min-max.
    - Restores urgency dominance: slack-based term is primary (0.55 weight), computed as normalized inverse slack *with floor-clipping* to avoid explosion.
    - Energy penalty now gated by *slack <= median_slack OR slack <= 0* (inclusive of urgent) and upward_rank > 70th percentile — ensures energy-awareness on all tight-slack tasks.
    - Fairness term uses *only* normalized wait_time (not wait_per_work) and activates only for non-urgent tasks with long wait — avoids starvation without over-penalizing small workloads.
    - Uncertainty is strictly a *modulator*: multiplies slack sensitivity and rank importance, never appears linearly — prevents dilution of deadline signals.
    - All norms include explicit degenerate-case fallbacks (zero-variance, singleton, empty).
    - Guarantees hard deadline enforcement: urgent tasks get -1e18 score (stronger than v1).
    '''
    eps = 1e-08
    min_exec_time = np.asarray(min_exec_time, dtype=float).copy()
    min_comm_time = np.asarray(min_comm_time, dtype=float).copy()
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float).copy()
    slack = np.asarray(slack, dtype=float).copy()
    upward_rank = np.asarray(upward_rank, dtype=float).copy()
    remaining_work = np.asarray(remaining_work, dtype=float).copy()
    ready_wait_time = np.asarray(ready_wait_time, dtype=float).copy()
    uncertainty = np.asarray(uncertainty, dtype=float).copy()
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=float)

    def robust_norm(x):
        if x.size == 0:
            return np.zeros_like(x)
        # Use MAD only if sufficiently large & variable; otherwise fallback to percentile min-max
        if N > 3:
            med = np.median(x)
            mad = np.median(np.abs(x - med))
            if mad > eps:
                scale = mad * 1.4826 + eps  # Consistent with std estimate
                normed = (x - med) / scale
                return np.clip(normed, -2.0, 2.0)
        # Fallback: percentile-clipped min-max
        p01 = np.percentile(x, 1.0)
        p99 = np.percentile(x, 99.0)
        x_clipped = np.clip(x, p01, p99)
        x_min = np.min(x_clipped)
        x_max = np.max(x_clipped)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        return (x_clipped - x_min) / (x_max - x_min + eps)

    is_urgent = (slack <= 0.0).astype(float)
    # Primary urgency signal: robust inverse slack, floor-clipped to prevent explosion
    abs_slack = np.abs(slack) + eps
    inv_slack = np.divide(1.0, abs_slack, out=np.zeros_like(slack), where=abs_slack != 0)
    inv_slack = np.clip(inv_slack, 0.01, 100.0)  # Strong floor/ceiling
    norm_inv_slack = robust_norm(inv_slack)

    # Critical-path importance: upward_rank modulated by uncertainty, then normalized
    risk_rank = upward_rank * (1.0 + 0.5 * uncertainty)
    norm_risk_rank = robust_norm(risk_rank)

    # Energy penalty gating: activate for urgent OR tight-slack (<= median) AND high-rank (>70th)
    median_slack = np.median(slack) if N > 1 else slack[0]
    tight_or_urgent_mask = ((slack <= median_slack) | is_urgent).astype(float)
    rank_threshold = np.percentile(upward_rank, 70.0) + eps
    high_rank_mask = (upward_rank > rank_threshold).astype(float)
    energy_gate = tight_or_urgent_mask * high_rank_mask

    # Risk-adjusted duration for energy density
    duration = min_exec_time + min_comm_time + eps
    risk_duration = duration * (1.0 + 0.3 * uncertainty)
    energy_density = np.divide(min_incremental_energy, risk_duration, out=np.zeros_like(min_incremental_energy), where=risk_duration != 0)
    norm_energy_density = robust_norm(energy_density)
    critical_energy_penalty = norm_energy_density * norm_risk_rank * energy_gate

    # Fairness: only non-urgent tasks with significant wait time (>= 90th percentile)
    wait_threshold = np.percentile(ready_wait_time, 90.0) if N > 1 else np.max(ready_wait_time)
    long_wait_mask = (ready_wait_time >= wait_threshold).astype(float)
    fairness_term = robust_norm(ready_wait_time) * (1.0 - is_urgent) * long_wait_mask

    # Uncertainty as pure modulator: boosts urgency and criticality signals, not standalone term
    uncertainty_mod = 1.0 + 0.4 * uncertainty
    modulated_urgency = norm_inv_slack * uncertainty_mod
    modulated_criticality = norm_risk_rank * uncertainty_mod

    # Final score composition: urgency dominates; energy penalty secondary; fairness tertiary
    score = np.full(N, 0.0, dtype=float)
    score = np.where(is_urgent, -1000000000000000000.0, score)
    score = np.where(is_urgent, score,
                     0.55 * modulated_urgency +
                     0.25 * critical_energy_penalty +
                     0.12 * fairness_term +
                     0.08 * modulated_criticality)

    # Final sanitization
    score = np.nan_to_num(score, nan=1000000000000000000.0, posinf=1000000000000000000.0, neginf=-1000000000000000000.0)
    score = np.clip(score, -1000000000000000000.0, 1000000000000000000.0)

    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
