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

    '\n    v5: Graded-violation-aware safety-energy priority with:\n      - Soft violation ranking: tanh(-slack/tau_viol) replaces hard mask → preserves relative urgency among violating tasks\n      - Monotonic slack gating: sigmoid(slack/tau_gate) used *only* for energy/criticality terms, not urgency\n      - Simplified CPP: upward_rank / (min_exec_time + min_comm_time + eps), scaled linearly by max(0, -slack) for violation pressure\n      - Strict fairness gating: wait penalty *only active when slack >= 0*, avoiding deadline conflict\n      - Uncertainty penalty *fully suppressed* when slack < 0 (violation dominates; uncertainty irrelevant)\n      - All normalization uses MAD-based robust scaling with explicit NaN/inf guard per term\n      - Deterministic, zero-safe, N=1 compliant, and strictly monotonic in slack for violation regime\n    '
    eps = 1e-08
    def sanitize(x):
        x = np.asarray(x, dtype=float)
        return np.nan_to_num(x, nan=eps, posinf=1e6, neginf=eps)
    min_exec_time = sanitize(min_exec_time)
    min_comm_time = sanitize(min_comm_time)
    min_incremental_energy = sanitize(min_incremental_energy)
    slack = sanitize(slack)
    upward_rank = sanitize(upward_rank)
    remaining_work = sanitize(remaining_work)
    ready_wait_time = sanitize(ready_wait_time)
    uncertainty = sanitize(uncertainty)
    
    def normalize_robust(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size == 1:
            return np.array([0.0])
        med = np.median(x)
        abs_dev = np.abs(x - med)
        mad = np.median(abs_dev)
        if mad > eps:
            scale = mad + eps
        else:
            xmin, xmax = np.min(x), np.max(x)
            scale = xmax - xmin
            if scale < eps:
                scale = eps
        normed = (x - med) / (scale + eps)
        return np.clip(normed, -5.0, 5.0)
    
    # Graded violation urgency: tanh(-slack/tau) → smoothly increases priority as slack decreases
    tau_viol = 0.4
    urgency_raw = np.tanh(-slack / tau_viol)
    norm_urgency = normalize_robust(urgency_raw)
    urgency_term = -6.0 * norm_urgency  # Strongest weight: violation urgency dominates
    
    # Critical-path pressure (CPP): upward_rank / latency, scaled by violation magnitude (linear penalty only when slack < 0)
    latency_cost = min_exec_time + min_comm_time + eps
    cpp_base = upward_rank / latency_cost
    norm_cpp = normalize_robust(cpp_base)
    violation_pressure = np.maximum(0.0, -slack)  # Only active when violating
    cpp_term = -2.0 * norm_cpp * (1.0 + 0.8 * normalize_robust(violation_pressure))
    
    # SEER (Safety-Energy-Efficiency Rank): (upward_rank * remaining_work) / energy, gated *only* by slack ≥ 0
    seer_base = upward_rank * remaining_work / (min_incremental_energy + eps)
    # Gating: only activate SEER when slack is non-negative (energy optimization only valid pre-violation)
    seer_gate = np.where(slack >= 0, 1.0 / (1.0 + np.exp(-(slack + eps)/0.6)), 0.0)
    seer_gated = seer_base * seer_gate
    norm_seer = normalize_robust(seer_gated)
    seer_term = -1.3 * norm_seer
    
    # Fairness: sqrt(wait) * exp(-uncertainty), but *strictly disabled* when slack < 0
    wait_safe = np.maximum(ready_wait_time, 0.0)
    sqrt_wait = np.sqrt(wait_safe + eps)
    exp_uncert = np.exp(-uncertainty)
    fairness_raw = np.where(slack >= 0, sqrt_wait * exp_uncert, 0.0)
    norm_fairness = normalize_robust(fairness_raw)
    fairness_term = -0.15 * norm_fairness
    
    # Uncertainty penalty: fully suppressed when slack < 0 (violation already dominant)
    uncert_penalty = np.where(slack >= 0, uncertainty * 0.1, 0.0)
    norm_uncert = normalize_robust(uncert_penalty)
    uncert_term = 0.4 * norm_uncert
    
    # Final score: sum of all terms
    score = urgency_term + cpp_term + seer_term + fairness_term + uncert_term
    
    # Final sanitization: clamp extremes, ensure finite values
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    score = np.clip(score, -1e12, 1e12)
    
    return score
