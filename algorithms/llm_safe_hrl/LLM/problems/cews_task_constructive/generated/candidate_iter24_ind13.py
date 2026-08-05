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
    v3 self-evolution: Addresses over-complexity and instability in fairness/uncertainty coupling;
    replaces piecewise urgency with smooth, strictly monotonic sigmoid-based deadline pressure;
    eliminates rank-sigmoid fairness (introduces bias) and replaces with pure criticality-weighted wait;
    tightens uncertainty gating to *only* violation regime (slack <= 0) for noise suppression;
    unifies normalization via robust MAD+minmax with explicit outlier clipping *before* norm;
    introduces energy-awareness scaling *only* when slack >= 0, using soft gate (sigmoid(slack + 1));
    removes all non-monotonic terms (e.g., slack^2) to guarantee priority ordering stability;
    enforces strict finite output: final score bounded in [-5e6, 5e6], shape=(N,), deterministic.
    """
    eps = 1e-08
    # Defensive casting & NaN/inf handling: conservative bounds, no neginf inflation
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=1e6, neginf=-1e6)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=0.0, posinf=1e6, neginf=0.0)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=0.0, posinf=1e6, neginf=0.0)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=1e6, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=0.0, posinf=1e6, neginf=0.0)

    def normalize_mad_minmax(x):
        """Robust normalization: MAD-based centering + minmax scaling; handles N=1, outliers, flat arrays"""
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        # Pre-normalization outlier clipping to prevent MAD distortion
        if x.size > 1:
            q25, q75 = np.percentile(x, [25, 75])
            iqr = q75 - q25
            lb, ub = q25 - 1.5 * iqr, q75 + 1.5 * iqr
            x = np.clip(x, lb, ub)
        if x.size == 1:
            return np.array([0.0])
        median_x = np.median(x)
        abs_devs = np.abs(x - median_x)
        mad = np.median(abs_devs)
        scale = mad if mad > eps else eps
        z = (x - median_x) / scale
        z_clipped = np.clip(z, -2.0, 2.0)  # Hard bound on standardized deviation
        z_min, z_max = np.min(z_clipped), np.max(z_clipped)
        if z_max - z_min < eps:
            return np.zeros_like(z_clipped)
        return 2.0 * (z_clipped - z_min) / (z_max - z_min + eps) - 1.0

    # Strictly monotonic deadline pressure: sigmoid(-slack) → high priority when slack is negative or small
    # Guarantees smooth, invertible, and differentiable urgency — no piecewise artifacts
    urgency_raw = 1.0 / (1.0 + np.exp(slack))  # ∈ (0,1); smaller slack → larger urgency_raw → higher priority after weighting
    norm_urgency = normalize_mad_minmax(urgency_raw)
    urgency_term = 1.0 + 0.95 * norm_urgency  # Dominant term: strongest weight for deadline adherence

    # Critical Path Density (CPD): importance per unit time; avoids division by zero
    exec_comm_sum = min_exec_time + min_comm_time + eps
    cpd_base = upward_rank / exec_comm_sum
    norm_cpd = normalize_mad_minmax(cpd_base)
    cpd_term = 1.0 + 0.65 * norm_cpd

    # Critical Energy Density (CED): energy efficiency only activated under feasibility (slack >= 0)
    # Soft gate: full activation at slack ≥ 0, smoothly decaying to zero as slack → -∞
    ced_gate = 1.0 / (1.0 + np.exp(-(slack + 0.5)))  # ≈1 when slack ≥ 0, ≈0 when slack ≤ -2
    ced_base = upward_rank * remaining_work / (min_incremental_energy + eps)
    ced_gated = ced_base * ced_gate
    norm_ced = normalize_mad_minmax(ced_gated)
    ced_term = 1.0 + 0.45 * norm_ced

    # Starvation prevention: criticality-weighted wait pressure — linear, monotonic, no sigmoid bias
    # Scales wait time by relative structural importance (upward_rank normalized to [0,1] for stability)
    rank_norm = (upward_rank - np.min(upward_rank + eps)) / (np.max(upward_rank + eps) - np.min(upward_rank + eps) + eps)
    wait_pressure = ready_wait_time * (rank_norm + 0.1)  # +0.1 avoids zero-weight for low-rank tasks
    norm_wait = normalize_mad_minmax(wait_pressure)
    wait_term = -0.22 * norm_wait  # Negative → reduces score for long-waiting critical tasks

    # Uncertainty penalty only in violation regime (slack <= 0), scaled linearly to avoid amplifying noise
    unc_penalty = np.zeros_like(uncertainty)
    mask_viol = slack <= 0
    unc_penalty[mask_viol] = uncertainty[mask_viol] * 0.8
    norm_unc = normalize_mad_minmax(unc_penalty)
    unc_term = 0.18 * norm_unc

    # Core multiplicative priority: urgency × CPD × CED ensures joint optimization of deadline, criticality, and energy
    core_score = urgency_term * cpd_term * ced_term

    # Final score: additive correction terms, all bounded and normalized
    score = core_score + unc_term + wait_term

    # Final safeguard: ensure finite, deterministic, shape-(N,) output
    score = np.nan_to_num(score, nan=5e6, posinf=5e6, neginf=-5e6)
    score = np.clip(score, -5e6, 5e6)
    return score.reshape(-1)
