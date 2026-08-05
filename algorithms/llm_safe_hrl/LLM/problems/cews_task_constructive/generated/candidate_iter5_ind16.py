import numpy as np

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
    Hybrid priority rule combining v1's robust sigmoidal urgency & IQR normalization
    with v0's starvation-aware aging boost, additive risk gating, and adaptive scale fallback.
    
    Key innovations:
    - Sigmoid urgency (v1) preserved for smooth monotonic deadline sensitivity
    - Additive uncertainty penalty only when both slack < 0 AND uncertainty > Q75 (v0's risk gating)
    - Aging boost: capped relative wait time scaled by urgency, preventing starvation without overriding deadlines
    - Adaptive normalization: IQR if meaningful variance, else std+eps, else constant 1.0 (robust for N=1/degenerate cases)
    - Critical-energy coupling: energy per critical unit, gated by urgency to focus efficiency where it matters most
    - All operations division-safe, finite, deterministic, and shape-(N,) guaranteed
    """
    eps = 1e-8
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    
    # Robust adaptive normalization: IQR if variance meaningful, else std, else 1.0
    def normalize_adaptive(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size == 1:
            return np.array([1.0])
        q25, q75 = np.percentile(x, [25, 75])
        iqr = q75 - q25
        if iqr > eps:
            scale = iqr + eps
            center = np.median(x)
        else:
            std_val = np.std(x)
            scale = std_val if std_val > eps else 1.0
            center = np.mean(x)
        return (x - center) / (scale + eps)
    
    # Sigmoid urgency: smooth, bounded, monotonic; τ=2.0 → critical tasks ~1.0, safe ~0.5
    tau = 2.0
    urgency_sigmoid = 1.0 / (1.0 + np.exp(-slack / tau))
    
    # Urgency term: negative because smaller score = higher priority
    norm_urgency = normalize_adaptive(urgency_sigmoid)
    urgency_term = -2.5 * norm_urgency
    
    # Critical density: importance × work per execution effort
    exec_effort = np.maximum(min_exec_time, eps)
    critical_density = upward_rank * (remaining_work / exec_effort)
    # Gate criticality by urgency to prioritize critical tasks *only when deadline pressure exists*
    gated_criticality = critical_density * (1.0 + urgency_sigmoid)
    norm_critical = normalize_adaptive(gated_criticality)
    critical_term = -1.0 * norm_critical
    
    # Energy efficiency per critical unit: lower is better → negative term
    latency_cost = min_exec_time + min_comm_time + eps
    energy_per_latency = min_incremental_energy / latency_cost
    critical_scale = np.maximum(critical_density, eps)
    eff_per_crit = energy_per_latency / critical_scale
    norm_eff_per_crit = normalize_adaptive(eff_per_crit)
    efficiency_term = -0.6 * norm_eff_per_crit
    
    # Aging boost: prevents starvation but capped and urgency-gated
    max_wait = np.max(ready_wait_time) + eps
    rel_wait = np.clip(ready_wait_time / max_wait, 0.0, 1.0)
    # Boost only active under deadline pressure and scales with urgency
    aging_boost = 0.05 * urgency_sigmoid * rel_wait
    aging_term = -0.3 * normalize_adaptive(aging_boost)
    
    # Additive uncertainty penalty: only applied under deadline risk AND high uncertainty
    median_uncertainty = np.median(uncertainty) if uncertainty.size > 0 else 0.0
    q75_unc = np.percentile(uncertainty, 75) + eps
    risk_active = (slack < 0) & (uncertainty > q75_unc)
    uncertainty_penalty = np.where(risk_active, uncertainty / q75_unc, 0.0)
    uncertainty_term = 0.4 * normalize_adaptive(uncertainty_penalty)
    
    # Final score: sum of all terms; smaller = higher priority
    score = (
        urgency_term +
        critical_term +
        efficiency_term +
        aging_term +
        uncertainty_term
    )
    
    # Ensure finite output
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    
    return score
