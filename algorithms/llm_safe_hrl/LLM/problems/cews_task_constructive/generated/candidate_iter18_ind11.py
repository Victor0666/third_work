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
    v2 evolution: Simplified, focused, and deadline-feasibility-aware.
    Key improvements:
    - Restores stronger energy-awareness: CED weight increased to -1.5 (vs -1.0 in v1) and unmasked for slack >= -tau_violation (soft gating).
    - Replaces CPD and separate uncertainty penalty with unified *criticality-pressure* term: 
      (upward_rank * (min_exec_time + min_comm_time)) * (1 + uncertainty), normalized and scaled only when slack > -0.5s (prevents over-penalizing near-violation).
    - Fairness term fully un-gated: sqrt(wait) / (1 + |slack| + eps) always active — ensures starvation prevention even under tight deadlines.
    - Uses tighter slack gating: tau_violation = 0.5s allows marginal energy optimization up to mild violation risk, improving tradeoff smoothness.
    - Removes redundant pressure scaling by urgency magnitude (caused noise); instead uses direct slack-aware activation.
    - All terms use adaptive normalization with explicit flat-array safety; final score bounded and finite.
    '''
    eps = 1e-08
    tau_violation = 0.5  # allow CED & pressure activation even for slightly negative slack (soft deadline boundary)
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    
    def normalize_adaptive(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size == 1:
            return np.array([0.0])
        q25, q75 = np.percentile(x, [25, 75])
        iqr = q75 - q25
        if iqr > eps:
            center = np.median(x)
            scale = iqr + eps
        else:
            xmin, xmax = (np.min(x), np.max(x))
            scale = xmax - xmin
            center = (xmax + xmin) / 2.0
            if scale < eps:
                scale = eps
        return (x - center) / (scale + eps)
    
    # Urgency: bounded, monotonic, zero-stable via arctan; normalized and weighted strongly
    tau_urgency = 1.0
    urgency_raw = np.arctan(-slack / tau_urgency)
    norm_urgency = normalize_adaptive(urgency_raw)
    urgency_term = -3.0 * norm_urgency  # increased weight for stricter deadline adherence
    
    # Critical-energy density: now activated for slack >= -tau_violation (soft feasibility boundary)
    ced_base = upward_rank * remaining_work / (min_incremental_energy + eps)
    ced_masked = np.where(slack >= -tau_violation, ced_base, 0.0)
    norm_ced = normalize_adaptive(ced_masked)
    ced_term = -1.5 * norm_ced  # restored stronger energy minimization where feasible
    
    # Unified criticality-pressure: latency-critical work × uncertainty, gated only for severe violation
    critical_work = min_exec_time + min_comm_time + eps
    pressure_base = upward_rank * critical_work * (1.0 + uncertainty)
    pressure_masked = np.where(slack > -tau_violation, pressure_base, 0.0)
    norm_pressure = normalize_adaptive(pressure_masked)
    pressure_term = 1.1 * norm_pressure  # slight boost to prioritize high-criticality, high-uncertainty tasks
    
    # Fairness: fully ungated — ensures aging signal always contributes to prevent starvation
    wait_safe = np.maximum(ready_wait_time, 0.0)
    sqrt_wait = np.sqrt(wait_safe + eps)
    fairness_scale = 1.0 + np.abs(slack) + eps
    fairness_raw = sqrt_wait / fairness_scale
    norm_fairness = normalize_adaptive(fairness_raw)
    fairness_term = -0.3 * norm_fairness  # increased weight vs v1 (-0.2) for stronger fairness
    
    # Assemble score: smaller = higher priority
    score = urgency_term + ced_term + pressure_term + fairness_term
    
    # Final numeric safeguard
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    
    return score
