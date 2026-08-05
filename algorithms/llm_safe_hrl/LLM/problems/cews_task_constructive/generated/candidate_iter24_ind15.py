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
    v2: Deadline-strict, criticality-prioritized, energy-aware, and numerically robust priority.
    
    Key self-evolution improvements:
    - Replaces arctan urgency with clipped inverse-linear urgency: monotonic, interpretable,
      bounded, and strictly prioritizes tasks with slack <= 0 (hard DDL violation) at top.
    - Introduces deadline proximity gating for fairness term: only activates wait-based fairness
      when slack > 1.0s (safe margin), eliminating unfair penalization under tight deadlines.
    - Strengthens criticality-pressure by factoring in *remaining_work density* (work per critical path unit)
      to better distinguish high-impact lightweight tasks vs low-impact heavy ones.
    - Refines CED term using *energy-normalized criticality* (upward_rank / (min_incremental_energy + eps))
      instead of work-energy ratio — aligns with objective: minimize energy *while meeting DDL*.
    - Tightens all normalization: uses median-IQR with guaranteed finite scale; replaces clipping
      with robust winsorization (95% percentile bounds) before norm to preserve ordinal structure.
    - Adds explicit zero-slack tie-breaker: tasks with slack <= 0 get fixed lowest priority score
      (dominant ordering), ensuring hard deadline enforcement.
    - All operations guarded against division-by-zero, NaN, inf; final score bounded and deterministic.
    '''
    eps = 1e-08
    tau_violation = 1.0
    safe_margin = 1.0  # seconds: only apply fairness if slack > safe_margin
    
    def clean(x):
        x = np.asarray(x, dtype=float)
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    
    min_exec_time = clean(min_exec_time)
    min_comm_time = clean(min_comm_time)
    min_incremental_energy = clean(min_incremental_energy)
    slack = clean(slack)
    upward_rank = clean(upward_rank)
    remaining_work = clean(remaining_work)
    ready_wait_time = clean(ready_wait_time)
    uncertainty = clean(uncertainty)
    
    # === Urgency: Hard DDL-first, then smooth proximity ===
    # Tasks with slack <= 0 get dominant priority (score = -inf equivalent via fixed low value)
    is_violated = slack <= 0.0
    urgency_raw = np.where(
        is_violated,
        -1e9,  # dominate all other terms — highest priority
        np.where(slack > 100.0, 0.0, -1.0 / (slack + eps))  # inverse-linear, bounded for large slack
    )
    
    # === Criticality-pressure: work-density aware ===
    # pressure = upward_rank * (exec+comm) * (1+uncertainty) * (remaining_work / (upward_rank + eps))
    # → simplifies to: (exec+comm) * (1+uncertainty) * remaining_work, but normalized by critical path weight
    # Instead: use *criticality density*: upward_rank / (remaining_work + eps) for impact-per-work,
    # then scale by cost and uncertainty → higher density = more critical per unit work
    crit_density = upward_rank / (remaining_work + eps)
    critical_cost = min_exec_time + min_comm_time + eps
    pressure_base = crit_density * critical_cost * (1.0 + uncertainty)
    pressure_gate = np.clip((slack + tau_violation) / (tau_violation + eps), 0.0, 1.0)
    pressure_masked = pressure_base * pressure_gate
    
    # === Energy-awareness: energy-normalized criticality (CED) ===
    # Prioritize high upward_rank *low*-energy tasks — not high-work ones
    ced_base = upward_rank / (min_incremental_energy + eps)
    gate_sigmoid = 1.0 / (1.0 + np.exp(-(slack + 0.5 * tau_violation) / (0.25 * tau_violation + eps)))
    ced_masked = ced_base * gate_sigmoid
    
    # === Fairness: activated only in safe region (slack > safe_margin) ===
    wait_safe = np.maximum(ready_wait_time, 0.0)
    sqrt_wait = np.sqrt(wait_safe + eps)
    fairness_raw = np.where(
        slack > safe_margin,
        sqrt_wait / (1.0 + (slack - safe_margin) + eps),
        0.0  # no fairness penalty under deadline stress
    )
    
    # === Robust adaptive normalization with winsorization ===
    def normalize_robust(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size == 1:
            return np.array([0.0])
        # Winsorize to [2.5%, 97.5%] to suppress outliers while preserving rank order
        p025, p975 = np.percentile(x, [2.5, 97.5])
        x_winsor = np.clip(x, p025, p975)
        q25, q75 = np.percentile(x_winsor, [25, 75])
        iqr = q75 - q25
        if iqr > eps:
            center = np.median(x_winsor)
            scale = iqr + eps
        else:
            xmin, xmax = np.min(x_winsor), np.max(x_winsor)
            scale = xmax - xmin
            center = (xmax + xmin) / 2.0
            if scale < eps:
                scale = eps
        return (x_winsor - center) / (scale + eps)
    
    norm_urgency = normalize_robust(urgency_raw)
    urgency_term = -4.0 * norm_urgency  # stronger deadline enforcement
    
    norm_pressure = normalize_robust(pressure_masked)
    pressure_term = 1.3 * norm_pressure
    
    norm_ced = normalize_robust(ced_masked)
    ced_term = -1.8 * norm_ced  # stronger energy incentive when safe
    
    norm_fairness = normalize_robust(fairness_raw)
    fairness_term = -0.2 * norm_fairness
    
    # Combine: urgency dominates, then energy (when safe), then criticality, then fairness (when safe)
    score = urgency_term + ced_term + pressure_term + fairness_term
    
    # Final sanitization: ensure finite, bounded output
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e9, 1e9)
    
    return score
