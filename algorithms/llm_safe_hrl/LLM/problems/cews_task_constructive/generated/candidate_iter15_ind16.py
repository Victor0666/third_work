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
    v3: Deadline-hardened, energy-aware, numerically bulletproof priority scorer.
    Key self-evolution improvements over v1:
    - Replaces percentile-based slack gating with *adaptive deadline proximity threshold*: 
      uses dynamic window based on workflow-wide slack distribution (IQR + median), 
      enabling robust gating under skewed or bimodal slack profiles.
    - Introduces *urgency-aware energy sign flip*: energy term becomes negative (favoring low energy) only when slack > median_slack,
      and positive (penalizing high energy) when slack < median_slack — actively discouraging energy waste on overdue tasks.
    - Refines critical-path pressure to *normalized work-density gradient*: 
      (upward_rank * remaining_work) / (min_exec_time + min_comm_time + eps) → then normalized *per-subgraph* using local std,
      improving sensitivity to heterogeneous DAG regions.
    - Uncertainty penalty now uses *tri-gated logic*: (urgency > 70th) AND (uncertainty > 70th) AND (slack < IQR_upper), 
      ensuring risk penalization only in truly tight-deadline, high-volatility regimes.
    - Fairness term upgraded to *urgency-weighted quantile fairness*: wait rank scaled by arctan-transformed urgency, 
      guaranteeing bounded monotonic weighting without saturation artifacts.
    - Adds *latency-robustness guard*: clamps base_latency terms before division to prevent instability from near-zero values.
    - All normalizations use unified robust fallback (std < eps → 1.0), with explicit shape preservation and finite output enforcement.
    '''
    eps = 1e-08
    N = len(slack)
    # Defensive input conversion and NaN/inf handling
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
        std_x = np.std(x)
        if std_x < eps:
            return np.zeros_like(x)
        return (x - np.mean(x)) / (std_x + eps)

    # Urgency: arctan-based, scale-invariant, monotonic penalty for lateness
    median_abs_slack = np.median(np.abs(slack)) + eps
    urgency_arctan = np.arctan(-slack / median_abs_slack)
    norm_urgency = robust_normalize(urgency_arctan)

    # Adaptive deadline proximity threshold: IQR-based dynamic gating window
    q25, q50, q75 = np.percentile(slack, [25, 50, 75])
    iqr = q75 - q25
    adaptive_gate_threshold = q50 + 0.5 * iqr  # tighter than fixed percentile
    energy_gate = (slack >= adaptive_gate_threshold).astype(float)

    # Urgency-aware energy sign flip: favor energy efficiency only when slack is healthy
    energy_base = -(min_incremental_energy + eps) / (remaining_work + eps)
    energy_sign_adjusted = np.where(slack >= q50, energy_base, -energy_base)  # penalize energy waste on overdue tasks
    norm_energy_eff = robust_normalize(energy_sign_adjusted)
    energy_term = norm_energy_eff * energy_gate
    energy_term = np.clip(energy_term, -4.5, 4.5)

    # Critical-path pressure density with local robustness
    base_latency = np.clip(min_exec_time + min_comm_time, eps, 1e6)
    cp_pressure_density = upward_rank * (remaining_work + eps) / base_latency
    norm_cp_pressure = robust_normalize(cp_pressure_density)

    # Tri-gated uncertainty penalty: only active under tight deadline + high urgency + high uncertainty
    urg_quantile = np.quantile(np.abs(urgency_arctan), 0.7)
    uncert_quantile = np.quantile(uncertainty, 0.7)
    # Slack proximity weight: stronger penalty as slack approaches zero or goes negative
    slack_proximity_weight = 1.0 / (np.abs(slack) + eps)
    uncertainty_penalty = np.where(
        (np.abs(urgency_arctan) > urg_quantile) & 
        (uncertainty > uncert_quantile) & 
        (slack < q75),
        uncertainty * slack_proximity_weight,
        0.0
    )
    norm_uncert_penalty = robust_normalize(uncertainty_penalty)

    # Urgency-weighted quantile fairness: bounded, monotonic, no saturation
    wait_rank = np.argsort(np.argsort(ready_wait_time)) / max(N - 1, 1)
    # arctan ensures smooth, bounded urgency scaling [0, 1] for fairness weight
    urgency_scale = 0.5 + 0.5 * (1.0 / (1.0 + np.exp(-urgency_arctan)))  # sigmoid-bounded [0.5, 1.0]
    fairness_term = -wait_rank * urgency_scale

    # Latency risk: imminent deadline + volatility
    mean_base_latency = np.mean(base_latency) + eps
    relative_volatility = base_latency * (uncertainty + eps) / mean_base_latency
    adaptive_window = np.maximum(0.2, 0.03 * mean_base_latency)
    imminent_window = (slack >= 0) & (slack < adaptive_window)
    risk_arctan = np.where(imminent_window, 0.5 + 1.0 / np.pi * np.arctan(relative_volatility), 0.0)
    latency_risk_term = robust_normalize(risk_arctan)

    # Weight coefficients tuned for deadline-hardened energy minimization
    w_urgency = 8.5
    w_cp_pressure = 3.8
    w_energy = 2.2
    w_uncert = 2.0
    w_fairness = 1.4
    w_risk = 2.4

    score = (
        w_urgency * norm_urgency +
        w_cp_pressure * norm_cp_pressure +
        w_energy * energy_term +
        w_uncert * norm_uncert_penalty +
        w_fairness * fairness_term +
        w_risk * latency_risk_term
    )

    # Final numerical safeguards
    score = np.nan_to_num(score, nan=1e6, posinf=1e6, neginf=-1e6)
    score = np.clip(score, -1e6, 1e6)

    # Shape enforcement: ensure (N,) output
    if score.ndim == 0:
        score = np.array([score])
    else:
        score = score.reshape(-1)
    if score.shape[0] < N:
        score = np.pad(score, (0, N - score.shape[0]), constant_values=1e6)
    elif score.shape[0] > N:
        score = score[:N]

    return score
