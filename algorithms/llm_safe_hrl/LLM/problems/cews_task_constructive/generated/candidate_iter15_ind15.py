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
    v2 refined priority rule: Fixes over-gating and conflation issues by:
    - Replacing linear-tail arctan with smooth, monotonic, *rank-preserving* urgency via tanh(-slack/tau),
      avoiding artificial flattening near deadlines and ensuring strict ordinal fidelity.
    - Removing median-based uncertainty gating → instead use *relative slack risk band*: 
      uncertainty penalty only activates when |slack| < 3*std(|slack|) OR |slack| < 1.0, capturing tight-deadline sensitivity without noise coupling.
    - Decoupling fairness from urgency: sqrt(wait) scaled *only* by normalized wait percentile (not sigmoid), 
      ensuring starvation relief is purely wait-driven and deterministic across slack regimes.
    - Strengthening critical-energy density (CED): now gated by *positive slack only*, and weighted by 
      upward_rank *normalized* to [0,1] via robust minmax to prevent dominance by outliers.
    - Introducing *latency-efficiency ratio*: (min_exec_time + min_comm_time) / (min_incremental_energy + eps)
      as a core term — directly prioritizes low-latency, low-energy tasks when feasible, aligned with DDL-hard objective.
    - All normalization uses adaptive IQR/minmax with explicit degenerate handling; no unbounded ops.
    """
    eps = 1e-08
    # Safe casting and nan/inf handling per input
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=eps, posinf=eps, neginf=eps)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=1e6, neginf=-1e6)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=eps, posinf=eps, neginf=eps)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

    def normalize_adaptive(x):
        """Robust normalization: IQR if non-degenerate, else minmax; size-1 → zero vector."""
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size == 1:
            return np.array([0.0])
        q25, q75 = np.percentile(x, [25, 75])
        iqr = q75 - q25
        if iqr > eps:
            center = np.median(x)
            scale = iqr + eps
        else:
            xmin, xmax = np.min(x), np.max(x)
            scale = xmax - xmin
            center = (xmax + xmin) / 2.0
            if scale < eps:
                scale = eps
        return (x - center) / (scale + eps)

    # === Urgency: tanh-based, rank-preserving, numerically stable ===
    # tanh(-slack/tau) ∈ (-1,1), strictly decreasing, no tail artifacts, preserves relative ordering even for sparse slack
    tau = 2.0
    urgency_raw = np.tanh(-slack / tau)
    norm_urgency = normalize_adaptive(urgency_raw)
    urgency_term = -3.2 * norm_urgency  # stronger pull for urgent tasks

    # === Critical-Energy Density (CED): slack-gated & outlier-robust ===
    # Upward rank normalized to [0,1] via adaptive minmax to prevent domination by outliers
    ur_norm = (upward_rank - np.min(upward_rank)) / (np.ptp(upward_rank) + eps) if len(upward_rank) > 1 else np.zeros_like(upward_rank)
    ur_norm = np.clip(ur_norm, 0.0, 1.0)
    ced_base = ur_norm * remaining_work / (min_incremental_energy + eps)
    # Gate *only* on positive slack (feasibility condition) — no "near-violation" energy optimization
    ced_masked = np.where(slack > 0.0, ced_base, 0.0)
    norm_ced = normalize_adaptive(ced_masked)
    ced_term = -1.4 * norm_ced  # prioritize energy-efficient execution *only* when deadline-safe

    # === Latency-Efficiency Ratio: favors fast + frugal tasks when feasible ===
    latency_eff_ratio = (min_exec_time + min_comm_time + eps) / (min_incremental_energy + eps)
    # Apply only when slack > 0 (no efficiency tradeoff under violation)
    latency_eff_masked = np.where(slack > 0.0, latency_eff_ratio, 1e6)  # large penalty if violated
    norm_latency_eff = normalize_adaptive(latency_eff_masked)
    latency_eff_term = 0.75 * norm_latency_eff

    # === Uncertainty Penalty: risk-aware but noise-robust ===
    # Activated only in high-risk bands: either very tight (|slack| < 1.0) OR relatively tight (|slack| < 3*std(|slack|))
    abs_slack = np.abs(slack) + eps
    std_abs_slack = np.std(abs_slack) if abs_slack.size > 1 else 0.0
    risk_band = (abs_slack < 1.0) | (abs_slack < 3.0 * std_abs_slack + eps)
    uncertainty_penalized = np.where(risk_band, uncertainty, 0.0)
    norm_uncertainty = normalize_adaptive(uncertainty_penalized)
    uncertainty_term = 0.20 * norm_uncertainty

    # === Fairness: pure wait-driven, decoupled from slack ===
    # sqrt(wait) scaled by percentile rank of wait time → ensures older tasks gain priority *proportionally*
    wait_safe = np.maximum(ready_wait_time, 0.0)
    sqrt_wait = np.sqrt(wait_safe + eps)
    # Percentile rank: 0.0 for min, 1.0 for max, linear interpolation
    if len(wait_safe) == 1:
        wait_rank = np.array([0.5])
    else:
        sorted_wait = np.sort(wait_safe)
        ranks = np.searchsorted(sorted_wait, wait_safe, side='left') / (len(wait_safe) - 1 + eps)
        wait_rank = np.clip(ranks, 0.0, 1.0)
    fairness_raw = sqrt_wait * (1.0 + wait_rank)  # stronger boost for both long wait *and* high relative rank
    fairness_clipped = np.clip(fairness_raw, 0.0, 0.4)
    norm_fairness = normalize_adaptive(fairness_clipped)
    fairness_term = -0.28 * norm_fairness

    # Combine terms: smaller score = higher priority
    score = urgency_term + ced_term + latency_eff_term + uncertainty_term + fairness_term
    # Final safeguard: finite values only
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    return score
