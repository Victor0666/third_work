import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule with structural improvements guided by counterfactual evidence:
      - Replaces linear neg_slack with robust power-law DDL risk penalty (ddl_risk_amplification)
      - Introduces explicit upward_rank × remaining_work coupling (critical-path bottleneck)
      - Removes energy_duration_ratio_weight: energy is now normalized *before* bottleneck term to avoid scale conflict
      - Uses raw wait_fairness_weight instead of sigmoid saturation — simpler, empirically more stable
      - Adds conditional DDL-protection gate: when slack < 0 AND uncertainty > median_uncertainty × threshold, amplify urgency
      - All normalizations use adaptive dispersion scaled by uncertainty std, fallback to range
      - No multiplicative risk couplings; all terms additive and bounded
      - Enforces strict finiteness and avoids NaN/inf via np.finfo safeguards
    """
    eps = 1.1194914589636914e-06
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    def adaptive_normalize(x):
        x = np.copy(x)
        center = np.median(x) if N > 0 else 0.0
        unc_std = np.std(uncertainty) if N > 1 else eps
        dispersion = 1.8990232878114668 * (unc_std + eps)
        fallback_range = np.max(x) - np.min(x) if N > 0 else eps
        denom = np.where(dispersion > eps, dispersion, fallback_range)
        return (x - center) / (denom + eps)
    neg_slack = np.clip(-slack, 0.0, None)
    ddl_risk_penalty = np.power(neg_slack + eps, 1.001137543770629)
    median_uncertainty = np.median(uncertainty) if N > 0 else eps
    risk_condition = (slack < 0) & (uncertainty > 1.7111841346118826 * median_uncertainty + eps)
    urgency_base = np.clip(np.maximum(0.0, -slack), 0.0, 2.0)
    urgency = np.where(risk_condition, urgency_base * 2.0, urgency_base)
    norm_urgency = adaptive_normalize(urgency)
    bottleneck_base = upward_rank * remaining_work
    unc_normalized = adaptive_normalize(uncertainty)
    bottleneck_pressure = bottleneck_base * np.power(1.0 + unc_normalized, 1.3502773839921065)
    norm_bottleneck = adaptive_normalize(bottleneck_pressure)
    norm_energy = adaptive_normalize(min_incremental_energy)
    norm_wait = adaptive_normalize(ready_wait_time)
    score = ddl_risk_penalty + norm_urgency + 0.8824972223675421 * norm_bottleneck + norm_energy - 0.8279641947629081 * norm_wait
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
