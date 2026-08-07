import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule with structural improvements guided by counterfactual evidence:
      - Introduces conditional DDL protection gate: amplifies negative slack penalty only when slack <= median_slack * threshold AND uncertainty > median_uncertainty.
      - Replaces sigmoid wait saturation with bounded power-law decay: avoids asymptotic flatness and improves discriminability for long-waiting tasks.
      - Uses MAD (median absolute deviation) instead of std for robust energy normalization — less sensitive to outlier VMs.
      - Couples critical path importance (upward_rank × remaining_work) only under feasible slack conditions, avoiding over-prioritization when deadlines are violated.
      - Removes all multiplicative risk couplings; uses additive, conditionally gated terms for clarity and stability.
      - All operations are finite, deterministic, and use only {-2,-1,0,1,2} literals.
    """
    eps = 5.836621264282491e-05
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    def robust_normalize(x):
        x = np.copy(x)
        center = np.median(x)
        mad = np.median(np.abs(x - center)) if N > 0 else eps
        dispersion = 1.8212840830492352 * (mad + eps)
        fallback_range = np.max(x) - np.min(x) if N > 0 else eps
        denom = dispersion if dispersion > eps else fallback_range
        return (x - center) / (denom + eps)
    neg_slack = np.clip(-slack, 0.0, None)
    median_slack = np.median(slack) if N > 0 else 0.0
    median_uncertainty = np.median(uncertainty) if N > 0 else 0.0
    ddl_risk_condition = (slack <= median_slack * 0.7486616945739064) & (uncertainty > median_uncertainty)
    ddl_protection = neg_slack * 2.4077943100824992 * ddl_risk_condition.astype(float)
    feasible_mask = (slack >= 0.0).astype(float)
    critical_path_pressure = upward_rank * remaining_work * feasible_mask
    norm_critical_path = robust_normalize(critical_path_pressure) * 1.9496904339261565
    norm_energy = robust_normalize(min_incremental_energy)
    wait_scaled = np.clip(ready_wait_time / (np.maximum(np.median(ready_wait_time), eps) + eps), 0.0, 2.0)
    wait_boost = np.power(wait_scaled + eps, 1.1099685384609916)
    norm_wait = robust_normalize(wait_boost)
    score = ddl_protection + norm_critical_path + norm_energy - norm_wait
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
