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
    v2 priority rule: Hard-deadline-dominant + urgency-preserving inverse-slack + 
                      criticality-gated energy efficiency + MAD-robust starvation relief +
                      simplified, monotonic risk coupling.

    Key improvements over v1:
    - Restores raw inverse-slack magnitude (no robust_mad_norm) for hard-deadline dominance;
      clips only to [0.01, 500] to preserve dynamic range and avoid suppression.
    - Replaces multi-gated wait pressure with single *urgency-aware* wait relief:
      only activates when slack < median_slack (i.e., risk-constrained), avoiding premature bias.
    - Simplifies energy penalty: uses upward_rank > median_upward *alone* as gate —
      removes fragile slack < median_slack conjunction that blocked low-energy options
      for tasks near but not below median slack.
    - Uncertainty coupling is now multiplicative *only on energy density*, not on slack or wait terms,
      preserving urgency fidelity and reducing over-smoothing.
    - Uses lightweight median/std fallback (no MAD recursion) for stability; all norms clipped to [-3,3].
    - Adds explicit "lateness penalty" for slack <= 0: linear penalty proportional to |slack|, unclipped,
      ensuring strict hard-deadline enforcement without relying solely on inverse-slack sign flip.
    - All divisions guarded; NaN/inf replaced deterministically; no side effects.
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

    # Lightweight robust normalization: median ± 3*std (MAD omitted for stability & speed)
    def robust_std_norm(x):
        x_clean = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x_clean.size == 0:
            return np.zeros_like(x_clean)
        med = np.median(x_clean)
        std = np.std(x_clean) + eps
        z = (x_clean - med) / std
        return np.clip(z, -3.0, 3.0)

    # 1. Hard deadline dominance: lateness penalty (unclipped, linear, high weight)
    lateness_penalty = np.where(slack <= 0.0, -1000000000000.0 + 1000.0 * np.abs(slack), 0.0)

    # 2. Urgency: raw inverse slack, confidence-weighted, clipped for stability — NOT normalized
    inv_slack_raw = np.divide(1.0, np.abs(slack) + eps, out=np.zeros_like(slack), where=np.abs(slack) + eps != 0)
    inv_slack_confidence = 1.0 / (1.0 + uncertainty + eps)
    inv_slack = inv_slack_raw * inv_slack_confidence
    inv_slack = np.clip(inv_slack, 0.01, 500.0)  # Preserve dynamic range for urgency ranking

    # 3. Criticality-latency proxy: duration × upward_rank, robustly normalized
    duration = min_exec_time + min_comm_time + eps
    crit_latency_proxy = duration * (1.0 + 0.5 * upward_rank)
    norm_crit_latency = robust_std_norm(crit_latency_proxy)

    # 4. Energy efficiency: risk-adjusted energy density, gated by criticality only
    energy_density = np.divide(min_incremental_energy, duration, out=np.zeros_like(min_incremental_energy), where=duration != 0)
    energy_density_adj = energy_density * (1.0 + uncertainty + eps)  # multiplicative risk coupling
    norm_energy_density = robust_std_norm(energy_density_adj)
    median_upward = np.median(upward_rank) + eps
    energy_gate = (upward_rank > median_upward).astype(np.float64)

    # 5. Starvation relief: urgency-aware wait pressure — only when slack is tight
    median_slack = np.median(slack) + eps
    wait_pressure_gate = (slack < median_slack).astype(np.float64)
    wait_per_work = np.divide(ready_wait_time, remaining_work + eps, out=np.zeros_like(ready_wait_time), where=remaining_work + eps != 0)
    norm_wait_per_work = robust_std_norm(wait_per_work)
    wait_relief = norm_wait_per_work * wait_pressure_gate

    # 6. Uncertainty bonus: lower priority for high-uncertainty tasks (encourages predictability)
    norm_uncertainty = robust_std_norm(uncertainty)

    # Final weighted score: smaller = higher priority
    # Ordering reflects hierarchy: lateness > urgency > criticality > energy > wait relief > uncertainty
    score = (
        lateness_penalty +              # dominates all; negative huge for late tasks
        0.3 * (-inv_slack) +            # urgency: higher inv_slack → more negative → higher priority
        0.25 * norm_crit_latency +      # critical path pressure
        0.2 * norm_energy_density * energy_gate +  # energy penalty only for high-rank tasks
        0.15 * (-wait_relief) +         # relieve starvation *only under deadline pressure*
        0.05 * norm_uncertainty         # mild penalty for high uncertainty (predictability bias)
    )

    # Final sanitization
    score = np.clip(score, -1000000000000.0, 1000000000000.0)
    score = np.nan_to_num(score, nan=1000000000000.0, posinf=1000000000000.0, neginf=-1000000000000.0)
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
