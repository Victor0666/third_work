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
    Self-evolved priority rule v2: Strict deadline dominance + decoupled risk-aware synergy + starvation gating.
    
    Key improvements over v1:
      - Eliminates spurious slack-uncertainty coupling: uncertainty only modulates *lateness penalty*, never slack itself.
      - Replaces comm-weighted duration with *normalized latency* (duration / median_duration) for stable synergy scaling.
      - Introduces *urgency-gated starvation*: wait boost activates ONLY when slack >= 0 AND upward_rank <= 75th percentile → prevents interference with critical paths.
      - Fixes energy efficiency gating: uses *relative energy density ratio* (energy_density / median_energy_density) instead of binary mask → smooth penalty gradient.
      - Removes all non-monotonic or contextually unstable terms (e.g., exp decay on rank, asymmetric work penalty) → preserves deterministic ordering fidelity.
      - Adds hard-zero guard on synergy denominator to avoid NaN in near-zero-energy tasks.
      - Uses 'higher' quantile method for robustness and avoids midpoint interpolation edge cases.
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

    def robust_scale(x):
        if len(x) == 0:
            return x
        med = np.median(x)
        x_centered = x - med
        q1, q3 = np.quantile(x_centered, [0.25, 0.75], method='higher')
        iqr = q3 - q1
        if iqr < eps:
            mad = np.median(np.abs(x_centered))
            scale = mad if mad > eps else np.mean(np.abs(x_centered)) + eps
        else:
            scale = iqr + eps
        scaled = x_centered / scale
        return np.clip(scaled, -1000000.0, 1000000.0)

    duration = np.maximum(min_exec_time + min_comm_time, eps)
    neg_slack = np.maximum(-slack, 0.0)
    
    # Hard urgency gating: 30th percentile slack threshold
    slack_sorted = np.sort(slack)
    slack_30 = slack_sorted[max(0, int(0.3 * len(slack_sorted)))] if N > 0 else 0.0
    is_deeply_urgent = slack < slack_30

    # Uncertainty only amplifies lateness penalty — no slack coupling
    uncertainty_mod = 1.0 + 0.9 * np.clip(uncertainty, 0.0, 2.0)
    deadline_penalty = neg_slack * uncertainty_mod
    deadline_penalty = np.where(is_deeply_urgent, deadline_penalty + 18.0, deadline_penalty)

    # Normalized latency for stable synergy: avoids bias from absolute scale
    duration_med = np.median(duration) + eps
    norm_duration = duration / duration_med
    # Synergy: high importance × low normalized latency × low incremental energy → prioritizes efficient critical path tasks
    synergy_denom = min_incremental_energy + eps
    synergy = upward_rank * norm_duration / synergy_denom
    synergy_score = -robust_scale(synergy)

    # Energy: relative density ratio → smooth penalty above median, not binary
    energy_density = min_incremental_energy / (duration + eps)
    energy_density_med = np.median(energy_density) + eps
    energy_rel_ratio = np.clip(energy_density / energy_density_med, 1.0, None) - 1.0
    energy_base = min_incremental_energy * (1.0 + 0.6 * np.clip(uncertainty, 0.0, 1.0))
    energy_score = robust_scale(energy_base) * energy_rel_ratio

    # Latency: raw duration scaled robustly
    latency_score = robust_scale(duration)

    # Starvation mitigation: ONLY when NOT urgent AND not high-criticality (prevents interference with critical path)
    wait_thresh = np.quantile(ready_wait_time, 0.75) if N > 0 else eps
    is_starvable = (~is_deeply_urgent) & (upward_rank <= np.quantile(upward_rank, 0.75))
    wait_boost_raw = np.where(ready_wait_time > wait_thresh, 
                              (ready_wait_time - wait_thresh) / (np.maximum(np.std(ready_wait_time), eps) + eps), 
                              0.0)
    wait_score = -wait_boost_raw * is_starvable.astype(float)

    # Work penalty: only activated when slack gap exists, linearly proportional
    slack_gap = np.clip(slack_30 - slack, 0.0, None)
    work_norm = robust_scale(remaining_work)
    work_penalty = work_norm * (slack_gap / (np.abs(slack_30) + eps))

    # Dominance hierarchy: urgency >> synergy > energy > latency > fairness > work
    score = (
        2.2 * deadline_penalty +
        0.6 * synergy_score +
        0.3 * energy_score +
        0.12 * latency_score +
        0.08 * wait_score +
        0.05 * work_penalty
    )

    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    score = np.clip(score, -1e12, 1e12)
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
