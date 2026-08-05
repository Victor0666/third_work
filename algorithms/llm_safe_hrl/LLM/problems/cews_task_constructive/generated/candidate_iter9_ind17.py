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
    Self-evolved priority rule: DDL-hardness + energy efficiency + fairness, with calibrated balance.
    Key improvements over v1:
      - Restores energy-per-work semantics (energy/remaining_work) for monotonic, interpretable efficiency signal;
      - Reduces hard-DDL offset to 120.0 → preserves urgency dominance *without* drowning energy fairness;
      - Replaces sigmoid+percentile coupling with linear urgency scaling: uncertainty penalty = uncertainty * (|slack|<tau ? 0 : norm_urgency),
        ensuring smooth, convex, and latency-aligned risk amplification;
      - Introduces *slack-aware energy gating*: energy term only activates when slack >= median_slack (not just >=0),
        preventing premature energy optimization before safety margin is confirmed;
      - Uses clipped tanh wait boost *without percentile*, gated by normalized slack margin and bounded by 0.3 → prevents unfair starvation relief;
      - All normalizations use robust IQR-based centering/scaling with explicit N=1 fallback and strict clipping [-2.5, 2.5];
      - Final score clamped to finite range and guaranteed deterministic.
    """
    eps = 1e-08
    # Safe casting & nan/inf handling
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
        if x.size == 1:
            return np.array([0.0])
        q25, q75 = np.percentile(x, [25, 75], overwrite_input=False)
        iqr = q75 - q25
        center = np.median(x, overwrite_input=False)
        if iqr > eps:
            scale = iqr
        else:
            std_val = np.std(x, ddof=0)
            scale = std_val if std_val > eps else np.mean(np.abs(x - center)) + eps
        norm = (x - center) / (scale + eps)
        return np.clip(norm, -2.5, 2.5)

    # --- 1. Urgency (DDL hardness): monotonic, bounded, additive dominance ---
    abs_slack = np.abs(slack)
    inv_slack = 1.0 / (abs_slack + eps)
    urgency_raw = np.where(slack < 0, inv_slack, 0.0)
    urgency_raw = np.clip(urgency_raw, 0.0, 500.0)  # tighter bound than v1
    norm_urgency = robust_normalize(urgency_raw)
    hard_ddl_offset = np.where(slack < 0, 120.0, 0.0)  # reduced from 200 → balances energy fairness

    # --- 2. Critical-path density: execution + comm latency weighted by importance ---
    latency_footprint = np.maximum(min_exec_time + min_comm_time, eps)
    critical_density = upward_rank * latency_footprint
    norm_critical_density = robust_normalize(critical_density)

    # --- 3. Energy efficiency: restored monotonic energy/remaining_work, context-gated ---
    energy_per_work = min_incremental_energy / (remaining_work + eps)
    # Gating: only prioritize energy when slack is safely above median (not just non-negative)
    median_slack = np.median(slack) + eps
    energy_gate = (slack >= median_slack).astype(float)
    norm_energy_eff = robust_normalize(energy_per_work) * energy_gate

    # --- 4. Uncertainty penalty: linear, convex, latency-aligned coupling ---
    # Activates only when slack < 0 AND uncertainty > median_uncert → no spurious risk amplification
    median_uncert = np.median(uncertainty) + eps
    risk_active = (slack < 0) & (uncertainty > median_uncert)
    # Scales linearly with normalized urgency (no sigmoid → preserves gradient alignment with lateness)
    uncertainty_penalty = np.where(risk_active, uncertainty * norm_urgency, 0.0)
    norm_uncert_penalty = robust_normalize(uncertainty_penalty)

    # --- 5. Fairness (aging): bounded, slack-gated tanh boost ---
    median_wait = np.median(ready_wait_time) + eps
    wait_tanh = np.tanh(ready_wait_time / (median_wait + eps))
    slack_margin = np.clip(slack, 0.0, 1e6)
    tau_safe = np.clip(np.mean(slack_margin) + eps, eps, 1e3)
    slack_safety_ratio = np.clip(slack_margin / (tau_safe + eps), 0.0, 1.0)
    wait_boost = wait_tanh * slack_safety_ratio
    wait_boost = np.clip(wait_boost, 0.0, 0.3)  # strict upper bound prevents dominance
    norm_wait_boost = robust_normalize(wait_boost)

    # --- 6. Base resource cost signals (light weighting) ---
    norm_exec = robust_normalize(min_exec_time)
    norm_comm = robust_normalize(min_comm_time)

    # --- Weighted combination: prioritizes urgency & critical path first, then energy, then fairness ---
    w_urgency = 5.5     # dominant but reduced from v1's 6.0 → allows energy to contribute meaningfully
    w_critical = 1.7    # slightly increased to emphasize path-criticality
    w_energy = 2.0      # restored full weight; now gated meaningfully
    w_uncert = 1.0      # unchanged: risk amplification is secondary
    w_wait = 0.4        # reduced from 0.6 → fairness is third priority
    w_exec = 0.1        # minimal direct exec-time penalty
    w_comm = 0.05       # minimal direct comm-time penalty

    score = (
        w_urgency * norm_urgency +
        w_critical * norm_critical_density +
        w_energy * norm_energy_eff +
        w_uncert * norm_uncert_penalty +
        w_wait * norm_wait_boost +
        w_exec * norm_exec +
        w_comm * norm_comm +
        hard_ddl_offset
    )

    # Final safeguard: ensure finite, deterministic, shape-(N,) output
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e9, 1e9)
    return score
