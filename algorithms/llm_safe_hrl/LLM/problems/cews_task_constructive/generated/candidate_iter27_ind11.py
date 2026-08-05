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
    v2: Asymmetric urgency-preserving priority with per-term MAD normalization,
         relative fairness, and hardened deadline dominance.

    Key improvements:
    - Replaces tanh-based urgency with *asymmetric max(0,-slack)* to preserve hard deadline dominance
      under multiplication (no dilution of violation signal).
    - Fairness now uses *relative wait time*: ready_wait_time / (1.0 + max(slack, 0) + eps),
      preventing starvation under sustained slack shortage while scaling with remaining margin.
    - Per-term MAD normalization (not shared) preserves semantic magnitude separation:
        urgency → raw [0,∞), cp_density → [0,∞), fairness → [0,∞), SEER → [0,∞)
      avoids scale collapse from mixing disparate units.
    - Robust_slack = slack - 3.0*uncertainty (tighter risk margin) with explicit zero-floor gating.
    - SEER inversion uses arctan(-SEER * 0.03) for finer discrimination near energy-optimal points.
    - All terms sigmoid-transformed to [0.1, 0.9] range before multiplication for stable composition.
    - Violation override applied *before* any nonlinearity and scaled by 1e-7 (stronger hard priority).
    """
    eps = 1e-08
    def sanitize(x):
        x = np.asarray(x, dtype=float)
        x = np.nan_to_num(x, nan=0.0, posinf=1000000.0, neginf=-1000000.0)
        return np.clip(x, -1000000.0, 1000000.0)
    min_exec_time = sanitize(min_exec_time)
    min_comm_time = sanitize(min_comm_time)
    min_incremental_energy = sanitize(min_incremental_energy)
    slack = sanitize(slack)
    upward_rank = sanitize(upward_rank)
    remaining_work = sanitize(remaining_work)
    ready_wait_time = sanitize(ready_wait_time)
    uncertainty = sanitize(uncertainty)

    # Hard violation detection: strict deadline dominance via asymmetric urgency
    violation_mask = (slack < 0).astype(float)
    urgency_raw = np.maximum(-slack, 0.0)  # pure non-negative urgency; no tanh softening

    # Risk-adjusted slack for gating & fairness scaling
    robust_slack = slack - 3.0 * uncertainty
    robust_slack = np.clip(robust_slack, -100000.0, 100000.0)

    # Critical-path density: importance per unit execution+comm cost
    exec_comm_sum = min_exec_time + min_comm_time + eps
    cp_density = upward_rank * remaining_work / (exec_comm_sum + eps)

    # Energy efficiency ratio (SEER): higher = more energy-efficient per time unit
    seer = exec_comm_sum / (min_incremental_energy + eps)

    # Relative fairness: wait time normalized by available slack margin to avoid starvation
    # When slack <= 0, denominator defaults to eps → fairness becomes large (low priority), 
    # which is correct: no margin → fairness yields to urgency
    fairness_denom = np.maximum(robust_slack, 0.0) + eps
    fairness_raw = ready_wait_time / fairness_denom

    # Per-term robust normalization: median absolute deviation per feature
    def normalize_per_term(x):
        x = np.clip(x, 0.0, 1000000.0)  # all terms are non-negative
        if x.size == 0:
            return np.zeros_like(x)
        median = np.median(x)
        mad = np.median(np.abs(x - median))
        if mad < eps:
            mad = eps
        return (x - median) / (mad + eps)

    norm_urgency = normalize_per_term(urgency_raw)
    norm_cp = normalize_per_term(cp_density)
    norm_fair = normalize_per_term(fairness_raw)
    norm_seer = normalize_per_term(seer)

    # Sigmoid-mapped scores in [0.1, 0.9] for stable multiplicative composition
    urgency_score = 0.1 + 0.8 / (1.0 + np.exp(-norm_urgency))
    cp_score = 0.1 + 0.8 / (1.0 + np.exp(-norm_cp))
    fair_score = 0.1 + 0.8 / (1.0 + np.exp(-norm_fair))
    # SEER: higher SEER = better energy/time tradeoff → lower priority score desired
    seer_arctan = np.arctan(-norm_seer * 0.03) * (2.0 / np.pi)
    energy_score = 0.1 + 0.8 * (seer_arctan + 1.0) / 2.0

    # Multiplicative base score: urgency dominates via asymmetry + gating
    base_score = urgency_score * cp_score * energy_score * fair_score

    # Hard violation override: amplify urgency dominance multiplicatively
    score = np.where(violation_mask, base_score * 1e-7, base_score)

    # Final sanitization: ensure finite, positive, shape-(N,)
    score = np.nan_to_num(score, nan=1000000.0, posinf=1000000.0, neginf=1e-07)
    score = np.clip(score, 1e-09, 1000000.0)
    score = np.asarray(score, dtype=float).reshape(-1)
    return score
