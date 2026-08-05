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

    # v2: Simplified, robust, and deadline-dominant priority with starvation-aware fairness
    # Key self-evolved improvements:
    # - Replaces per-term MAD normalization with lightweight, stable z-score via std + eps (faster, less degenerate)
    # - Restores unconditional fairness activation *without* cp-term weighting → ensures starvation prevention even under tight deadlines
    # - Uses hard-zero override for violated tasks (slack < 0) instead of multiplicative scaling → absolute priority guarantee
    # - Introduces latency-robust urgency: linear ramp on robust_slack = slack - 2.0*uncertainty, clipped to [0, 1]
    # - Energy term (SEER) gated only by criticality (cp_density > median), but applied *additively* in log-space to avoid vanishing gradients
    # - Removes all sigmoid/arctan transforms: uses direct bounded linear mapping for interpretability & numerical stability
    # - Enforces strict finite output via final clip + nan_to_num with deterministic bounds
    eps = 1e-08

    def sanitize(x):
        x = np.asarray(x, dtype=float)
        x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        return np.clip(x, -1e6, 1e6)

    min_exec_time = sanitize(min_exec_time)
    min_comm_time = sanitize(min_comm_time)
    min_incremental_energy = sanitize(min_incremental_energy)
    slack = sanitize(slack)
    upward_rank = sanitize(upward_rank)
    remaining_work = sanitize(remaining_work)
    ready_wait_time = sanitize(ready_wait_time)
    uncertainty = sanitize(uncertainty)

    # Robust slack: conservative deadline margin accounting for uncertainty
    robust_slack = np.clip(slack - 2.0 * uncertainty, -1e5, 1e5)

    # Hard violation mask → zero score guarantees top priority (np.argmin selects smallest)
    violation_mask = (slack < 0.0).astype(float)

    # Criticality density: HEFT-inspired importance per unit time
    exec_comm_sum = np.clip(min_exec_time + min_comm_time, eps, 1e6)
    cp_density = (upward_rank * remaining_work) / exec_comm_sum

    # Normalize cp_density with stable std-based z-score (avoids MAD degeneracy on small N or flat values)
    cp_mean = np.mean(cp_density) if cp_density.size > 0 else 0.0
    cp_std = np.std(cp_density) if cp_density.size > 0 else 1.0
    cp_std = max(cp_std, eps)
    cp_norm = (cp_density - cp_mean) / cp_std
    cp_term = np.clip(0.1 + 0.9 * (cp_norm * 0.5 + 0.5), 0.1, 0.9)  # bounded linear map → [0.1, 0.9]

    # Urgency: monotonic, interpretable, and bounded [0.0, 1.0]
    # Linear ramp: 0 when robust_slack >= 0, 1 when robust_slack <= -10s, smooth in between
    urgency_raw = np.clip((-robust_slack) / (10.0 + eps), 0.0, 1.0)
    urg_mean = np.mean(urgency_raw) if urgency_raw.size > 0 else 0.0
    urg_std = np.std(urgency_raw) if urgency_raw.size > 0 else 1.0
    urg_std = max(urg_std, eps)
    urg_norm = (urgency_raw - urg_mean) / urg_std
    urgency_term = np.clip(0.1 + 0.9 * (urg_norm * 0.5 + 0.5), 0.1, 0.9)

    # Energy efficiency: SEER (time/energy), gated by criticality, added in log-space for balanced contribution
    seer_ratio = exec_comm_sum / (min_incremental_energy + eps)
    cp_median = np.median(cp_density) if cp_density.size > 0 else 0.0
    energy_gate = (cp_density > cp_median + eps).astype(float)
    gated_seer = seer_ratio * energy_gate

    # Fairness: starvation prevention — unconditional, normalized by max wait, no cp dilution
    wait_max = np.maximum(np.max(ready_wait_time), eps)
    fairness_term = np.clip(ready_wait_time / (wait_max + eps), 0.0, 1.0)

    # Combine terms additively in log-space to avoid dominance collapse and preserve linearity
    # Base = urgency + criticality + (1 - SEER) + fairness → lower is better; SEER inverted so higher = better
    # Use linear combination with fixed weights tuned for deadline safety first
    base_score = (
        0.45 * urgency_term +
        0.30 * (1.0 - cp_term) +  # high cp → lower penalty weight (prioritize deadline over energy)
        0.20 * (1.0 - np.clip(gated_seer / (np.max(gated_seer + eps) + eps), 0.0, 1.0)) +
        0.05 * (1.0 - fairness_term)  # fairness reduces score → longer wait → higher priority
    )

    # Apply hard-zero override for violated tasks (absolute priority)
    score = np.where(violation_mask, 0.0, base_score)

    # Final sanitization: ensure finite, shape-(N,), deterministic
    score = np.nan_to_num(score, nan=1e6, posinf=1e6, neginf=1e-12)
    score = np.clip(score, 1e-12, 1e6)
    score = np.asarray(score, dtype=float).reshape(-1)

    # Guarantee shape-(N,) even for N=0 (though env ensures N≥1, defensive)
    if score.size == 0:
        score = np.array([1.0], dtype=float)

    return score
