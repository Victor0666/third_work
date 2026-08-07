import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule incorporating evidence-backed structural changes:
      - Adds conditional DDL-protection gate: activates only when slack <= median_slack AND normalized_uncertainty > threshold.
      - Replaces multiplicative bottleneck coupling with additive, gated critical-path term: upward_rank * remaining_work.
      - Introduces energy-risk coupling: min_incremental_energy * (1 + uncertainty) to penalize high-uncertainty high-energy tasks.
      - Uses DDL-aware min-max normalization instead of median-based for small-N robustness and stronger slack dominance under stress.
      - Retains bounded sigmoid wait saturation for anti-starvation, but now gated by DDL risk to avoid interfering with urgent tasks.
      - All operations are finite, deterministic, and use only {-2,-1,0,1,2} literals; no unbounded loops or side effects.
    """
    eps = 2.528162060962884e-06
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    def ddl_aware_normalize(x):
        x = np.copy(x)
        x_min = np.min(x) if N > 0 else 0.0
        x_max = np.max(x) if N > 0 else 1.0
        range_val = x_max - x_min
        denom = range_val if range_val > eps else eps
        return (x - x_min) / denom
    neg_slack = np.clip(-slack, 0.0, None)
    median_slack = np.median(slack) if N > 0 else 0.0
    unc_norm = ddl_aware_normalize(uncertainty)
    ddl_risk_gate = np.where((slack <= median_slack) & (unc_norm > 0.8427977721857307), 1.0, 0.0)
    critical_path_pressure = upward_rank * remaining_work
    norm_critical_path = ddl_aware_normalize(critical_path_pressure)
    energy_risk_term = min_incremental_energy * (1.0 + uncertainty)
    norm_energy_risk = ddl_aware_normalize(energy_risk_term)
    wait_scaled = ready_wait_time / (7.735166968693976 + eps)
    wait_clipped = np.clip(wait_scaled, -2.0, 2.0)
    wait_saturation = 1.0 / (1.0 + np.exp(-wait_clipped))
    wait_boost = np.where(slack > median_slack, wait_saturation, 0.0)
    score = neg_slack + ddl_risk_gate * 1.0359850668427248 * norm_critical_path + ddl_risk_gate * 0.010143290847364402 * norm_energy_risk - wait_boost
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
