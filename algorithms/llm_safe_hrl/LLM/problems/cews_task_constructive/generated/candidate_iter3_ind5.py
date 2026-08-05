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
    Self-evolved priority rule: restores ordinal urgency for deadline risk,
    decouples criticality & energy scaling, and sharpens uncertainty gating.
    
    Key improvements:
    - Slack penalty retains raw exponential ordering for late tasks (no [0,1] compression)
    - Criticality-energy coupling uses *calibrated* scaling: upward_rank normalized separately,
      work/energy efficiency scaled by median-based robust ratio to preserve relative magnitude
    - Uncertainty gating simplified: active only when slack < 0 (hard violation) OR slack < 30s (soft urgency)
    - Waiting boost uses relative percentile rank instead of max-based sigmoid → more robust to outliers
    - All features re-normalized via IQR but kept in original units where semantics matter (e.g., slack penalty)
    - Final score strictly bounded and finite; no nan/inf propagation.
    """
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
    
    # Robust normalization helper: preserves sign, handles flat arrays
    def normalize_robust(x):
        q75, q25 = np.percentile(x, [75, 25], method='midpoint')
        iqr = q75 - q25 + eps
        med = np.median(x)
        return (x - med) / iqr
    
    # --- 1. Deadline urgency: preserve ordinal penalty for lateness ---
    # For slack < 0: exp(-slack) — larger penalty for more negative slack (strict ordering)
    # For slack >= 0: smooth decay ~ 1/(1+slack), capped at 1.0 → avoids discontinuity & rewards margin
    slack_penalty = np.where(
        slack < 0,
        np.clip(np.exp(-slack), 1.0, 25000.0),
        np.clip(1.0 / (1.0 + 0.1 * slack), 0.01, 1.0)
    )
    
    # --- 2. Criticality-energy coupling: calibrated, not joint-normalized ---
    # Upward rank: normalize separately to avoid scale collapse
    norm_upward = normalize_robust(upward_rank)
    # Work/energy efficiency: robust ratio scaled to [0,1] range using median reference
    efficiency = remaining_work / (min_incremental_energy + eps)
    median_eff = np.median(efficiency) + eps
    scaled_efficiency = np.clip(efficiency / median_eff, 0.1, 10.0)  # prevents extreme outliers
    # Coupled score: product, then robustly normalized to maintain interpretability
    critical_energy_score = norm_upward * scaled_efficiency
    norm_critical_energy = normalize_robust(critical_energy_score)
    
    # --- 3. Starvation prevention: percentile-based waiting boost ---
    # Use percentile rank (0–1) instead of max-scaled sigmoid → robust to outliers
    wait_percentile = np.argsort(np.argsort(ready_wait_time)) / (N - 1 + eps) if N > 1 else np.array([0.0])
    wait_boost = 0.35 * (1.0 / (1.0 + np.exp(-(wait_percentile * 10.0 - 5.0))))
    
    # --- 4. Uncertainty gating: tightened to deadline-critical contexts ---
    # Active only when slack is tight (< 0 or < 30s) → focuses on high-risk scenarios
    tight_slack_mask = (slack < 0) | (slack < 30.0)
    # Normalize uncertainty only under tight condition; else zero
    norm_uncertainty = np.where(tight_slack_mask, normalize_robust(uncertainty), 0.0)
    
    # --- 5. Base feature scores (all robustly normalized) ---
    norm_exec = normalize_robust(min_exec_time)
    norm_comm = normalize_robust(min_comm_time)
    norm_energy = normalize_robust(min_incremental_energy)
    
    # --- 6. Weighted linear combination: prioritizes deadline safety first ---
    # Higher weight on slack_penalty (urgency) and critical_energy (efficiency under constraint)
    score = (
        0.1 * norm_exec +
        0.08 * norm_comm +
        0.15 * norm_energy -
        0.4 * slack_penalty +  # Dominant negative term: lower score = higher priority for urgent tasks
        0.0 * norm_upward +     # Already embedded in critical_energy_score
        0.0 * scaled_efficiency +  # Ditto
        -0.25 * norm_critical_energy +  # Negative: higher critical-energy score → lower priority (we prefer efficient critical work)
        0.07 * wait_boost -
        0.05 * norm_uncertainty
    )
    
    # Ensure finite output: clamp extremes, replace nan/inf
    return np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
