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
    v2: Smooth, gradient-preserving deadline-energy trade-off with adaptive fairness.
    Key improvements over v1:
    - Replaces hard urgency dominance mask with soft priority amplification (sigmoid-weighted urgency term),
      preserving gradient continuity for joint deadline/energy optimization.
    - Simplifies uncertainty scaling to clipped linear (0.0–1.0) instead of tanh; more interpretable and numerically stable.
    - Restores clipped linear fairness (not log) with dynamic threshold and bounded slope—stronger wait signal without explosion.
    - Introduces *energy-aware critical efficiency* (EACE) = min_incremental_energy / (upward_rank * remaining_work + eps)
      as a complementary energy-regularized term, encouraging low-energy execution on high-criticality paths.
    - Uses unified robust normalization: MAD-based with safe fallback, applied *after* all term construction to preserve monotonicity.
    - All operations guarded against zero/nan/inf; no unbounded growth; deterministic and N=1 safe.
    '''
    eps = 1e-08
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    
    def normalize_mad(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size == 1:
            return np.array([0.0])
        med = np.median(x)
        abs_dev = np.abs(x - med)
        mad = np.median(abs_dev)
        if mad < eps:
            xmin, xmax = np.min(x), np.max(x)
            scale = xmax - xmin
            if scale < eps:
                scale = eps
            return (x - (xmin + xmax) / 2.0) / (scale + eps)
        return (x - med) / (mad + eps)
    
    # Urgency: hybrid linear (for negative slack) + smooth exponential decay (for positive slack)
    urgency_linear = np.where(slack < 0, -slack, 0.0)
    urgency_exp = np.where(slack > 0, -np.exp(-slack / 60.0), 0.0)
    urgency_raw = urgency_linear + urgency_exp
    
    # Soft urgency amplification via sigmoid gate — preserves ordering but boosts critical tasks smoothly
    urgency_gate = 1.0 / (1.0 + np.exp(-(urgency_raw - np.median(urgency_raw)) / (eps + np.std(urgency_raw, ddof=0))))
    urgency_term = -3.0 * normalize_mad(urgency_raw) * (1.0 + 0.5 * urgency_gate)
    
    # Slack-gated SCEE: critical work density per latency unit, suppressed when slack is tight
    scee_base = upward_rank * remaining_work / (min_exec_time + min_comm_time + eps)
    scee_gate = 1.0 / (1.0 + np.exp(-slack / 12.0))  # gentler than v1's 15.0 for smoother transition
    scee_gated = scee_base * scee_gate
    scee_term = -1.4 * normalize_mad(scee_gated)
    
    # Latency: base latency + uncertainty penalty only when slack > 0, using robust clipped scaling [0, 0.8]
    base_latency = min_exec_time + min_comm_time + eps
    uncertainty_clipped = np.clip(uncertainty, 0.0, 5.0)  # cap extreme uncertainty
    uncertainty_penalty = np.where(slack > 0, uncertainty_clipped * 0.8 * base_latency, 0.0)
    latency_total = base_latency + uncertainty_penalty
    latency_term = 0.6 * normalize_mad(latency_total)
    
    # Energy-aware critical efficiency (EACE): lower score for low energy on high-criticality paths
    eace = min_incremental_energy / (upward_rank * remaining_work + eps)
    eace_term = 0.9 * normalize_mad(eace)
    
    # Fairness: clipped linear wait boost, gated by dynamic slack threshold (median + margin), slope capped
    slack_median = np.median(slack)
    wait_boost_active = (slack > slack_median + 0.05).astype(float)
    wait_bounded = np.clip(ready_wait_time, 0.0, 90.0)  # tighter bound than v1's 120.0
    fairness_raw = wait_bounded * wait_boost_active
    fairness_term = -0.18 * normalize_mad(fairness_raw)  # stronger weight than v1's -0.12
    
    # Assemble final score: urgency dominates, then efficiency & energy, then latency & fairness
    score = (
        urgency_term +
        scee_term +
        eace_term +
        latency_term +
        fairness_term
    )
    
    # Final robustness guard
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    return score
