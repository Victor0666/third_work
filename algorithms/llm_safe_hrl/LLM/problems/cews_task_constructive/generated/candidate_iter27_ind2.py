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
    v5 priority rule: Hard feasibility dominance + linear urgency amplification + 
                      criticality-weighted risk-energy density + 
                      starvation-resilient wait-pressure + 
                      uncertainty-robust slack scaling.

    Key improvements over v4:
    - Restores strong, interpretable linear urgency: violation penalty = -slack (clipped), tight slack reward = (q50 - slack)/duration → preserves discriminability near deadline.
    - Removes fragile non-linear gates (sigmoid, exp) that blurred urgency hierarchy; replaces with piecewise-linear, numerically stable scaling.
    - Simplifies energy term to criticality-weighted risk-energy density: (energy / duration) * (1 + uncertainty) * upward_rank → directly penalizes high-energy, high-risk, high-criticality tasks.
    - Broadens starvation rescue: lowers wait_ratio threshold to >1.2 (not >2.0), retains slack < 300s and upward_rank >= median + 0.3*IQR → improves feasible_seed_rate without compromising robustness.
    - Introduces uncertainty-robust slack scaling: slack_reward = max(0, min(1, (slack - q10)/(q90 - q10 + eps))) → monotonic, bounded, avoids division-by-zero and outlier sensitivity.
    - Adds duration-normalized criticality penalty only for long tasks (duration > 0.7*max_duration) to avoid over-penalizing short critical tasks.
    - All normalization uses robust IQR z-clipping (±6σ); weights sum to 1.0: 0.50 (urgency) + 0.25 (risk-energy) + 0.12 (criticality-penalty) + 0.08 (starvation) + 0.05 (uncertainty).
    """
    eps = 1e-08
    min_exec_time = np.asarray(min_exec_time, dtype=np.float64).copy()
    min_comm_time = np.asarray(min_comm_time, dtype=np.float64).copy()
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=np.float64).copy()
    slack = np.asarray(slack, dtype=np.float64).copy()
    upward_rank = np.asarray(upward_rank, dtype=np.float64).copy()
    remaining_work = np.asarray(remaining_work, dtype=np.float64).copy()
    ready_wait_time = np.asarray(ready_wait_time, dtype=np.float64).copy()
    uncertainty = np.asarray(uncertainty, dtype=np.float64).copy()
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=np.float64)

    def robust_zclip(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.clip(x, -1000000000000.0, 1000000000000.0)
        if N == 1:
            return np.array([0.0])
        q25, q50, q75 = np.quantile(x, [0.25, 0.5, 0.75], method='midpoint')
        iqr = q75 - q25 + eps
        z = (x - q50) / iqr
        return np.clip(z, -6.0, 6.0)

    task_duration = np.maximum(min_exec_time + min_comm_time, eps)
    violated_mask = slack < 0
    feasibility_boost = np.full(N, 0.0, dtype=np.float64)
    feasibility_boost[violated_mask] = -1000000000000.0

    # Linear urgency: strict violation penalty, smooth slack reward
    urgency_base = np.zeros_like(slack)
    urgency_base[violated_mask] = np.clip(-slack[violated_mask], 0.0, 1000.0)
    if N > 1:
        q10, q50, q90 = np.quantile(slack, [0.1, 0.5, 0.9], method='midpoint')
        slack_range = q90 - q10 + eps
        # Monotonic, bounded slack reward: 0→1 over [q10, q90]
        slack_reward = np.clip((slack - q10) / slack_range, 0.0, 1.0)
        tight_mask = ~violated_mask & (slack < q50)
        urgency_base[tight_mask] = np.clip((q50 - slack[tight_mask]) / (task_duration[tight_mask] + eps), 0.0, 6.0)
        relaxed_mask = ~violated_mask & (slack >= q50)
        urgency_base[relaxed_mask] = np.clip((1.0 - slack_reward[relaxed_mask]) * 2.0, 0.0, 2.0)
    else:
        urgency_base[~violated_mask] = 1.0

    # Criticality-weighted risk-energy density: energy per duration × risk × criticality
    risk_energy_density = np.divide(min_incremental_energy, task_duration, out=np.zeros_like(min_incremental_energy), where=task_duration != 0)
    risk_energy_density = np.nan_to_num(risk_energy_density, nan=0.0, posinf=0.0, neginf=0.0)
    risk_energy_density = risk_energy_density * (1.0 + uncertainty) * upward_rank

    # Duration-normalized criticality penalty: only for long tasks (>70% max duration)
    if N > 1:
        max_duration = np.max(task_duration) + eps
        long_task_mask = task_duration > 0.7 * max_duration
        duration_penalty = np.where(long_task_mask, (task_duration / max_duration) * upward_rank, 0.0)
    else:
        duration_penalty = np.zeros_like(upward_rank)

    # Starvation rescue: broader coverage (wait_ratio > 1.2, not >2.0), same criticality threshold
    if N > 1:
        ur_median = np.median(upward_rank)
        ur_q25, ur_q75 = np.quantile(upward_rank, [0.25, 0.75], method='midpoint')
        ur_iqr = ur_q75 - ur_q25 + eps
        crit_threshold = ur_median + 0.3 * ur_iqr
    else:
        crit_threshold = upward_rank[0]
    wait_ratio = np.divide(ready_wait_time, task_duration, out=np.zeros_like(ready_wait_time), where=task_duration != 0)
    wait_ratio = np.nan_to_num(wait_ratio, nan=0.0, posinf=0.0, neginf=0.0)
    is_starvable = (wait_ratio > 1.2) & (upward_rank >= crit_threshold) & (slack < 300.0)
    starvation_rescue = np.where(is_starvable, wait_ratio * (0.5 + 0.5 * np.clip(upward_rank, 0.0, 1.0)), 0.0)
    starvation_rescue = np.clip(starvation_rescue, 0.0, 3.0)

    # Normalize components independently
    norm_urgency = robust_zclip(urgency_base)
    norm_energy = robust_zclip(risk_energy_density)
    norm_penalty = robust_zclip(duration_penalty)
    norm_starvation = robust_zclip(starvation_rescue)
    norm_uncertainty = robust_zclip(uncertainty)

    # Weighted linear combination (weights sum to 1.0)
    score = (0.50 * norm_urgency + 
             0.25 * norm_energy + 
             0.12 * norm_penalty + 
             0.08 * norm_starvation + 
             0.05 * norm_uncertainty)

    score = score + feasibility_boost
    score = np.nan_to_num(score, nan=1000000000000.0, posinf=1000000000000.0, neginf=-1000000000000.0)
    score = np.clip(score, -1000000000000.0, 1000000000000.0)
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
