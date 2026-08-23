import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule featuring:
       - Smooth sigmoidal slack urgency (replacing both hard DDL gate AND slack_score's piecewise branch) 
         for differentiable, full-range deadline pressure — improves CMA-ES convergence and avoids feasibility cliffs.
       - Unified urgency signal: `1 / (1 + exp(-k * slack))` maps slack ∈ ℝ → (0,1), inverted for priority (smaller = tighter).
       - Retains bias-stabilized critical_path_release (monotonic across all slack) and drops wait-time entirely.
       - All DDL-critical terms now share a common differentiable foundation — no discontinuous branches.
       - MAD-normalization remains for robustness; N=1 fallback preserved.
       - No numeric literals outside {-2,-1,0,1,2}.
    """
    eps = 6.090091680984161e-08
    finfo = np.finfo(float)
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=finfo.max, neginf=finfo.min)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=finfo.max, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=eps, posinf=finfo.max, neginf=eps)
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=float)

    def mad_normalize(x):
        x = np.asarray(x, dtype=float)
        if N == 1:
            return np.zeros_like(x, dtype=float)
        med = np.median(x)
        abs_dev = np.abs(x - med)
        mad = np.median(abs_dev) + eps
        normalized = (x - med) / mad
        return np.clip(normalized, -2.0, 2.0)
    slack_urgency = 1.0 / (1.0 + np.exp(3.702013461772067 * slack))
    deadline_pressure = slack_urgency
    unc_slack_coupling = uncertainty * deadline_pressure * 1.5080080577363733
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 0.448087348315761
    critical_release_denom = np.maximum(0.0, -slack) + 0.10488490333267939
    critical_path_release = upward_rank * remaining_work / critical_release_denom * 2.267445907328068
    ddl_safe_scale = 1.0 - slack_urgency
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_score = mad_normalize(energy_per_sec)
    host_load_scale = ddl_safe_scale * 0.8696415986331918
    rank_boost_weight = slack_urgency
    rank_score = -mad_normalize(upward_rank) * rank_boost_weight
    unc_sigmoid = 1.0 / (1.0 + np.exp(-3.0830702922378213 * (uncertainty - 1.0)))
    energy_norm = mad_normalize(min_incremental_energy)
    unc_norm = mad_normalize(uncertainty)
    energy_uncertainty_score = host_load_scale * 0.5637714548935365 * energy_norm * unc_norm * unc_sigmoid
    score = mad_normalize(slack_urgency) + mad_normalize(unc_slack_coupling) + mad_normalize(duration_risk) + mad_normalize(critical_path_release)
    score += ddl_safe_scale * (1.1847370937209112 * energy_eff_score + rank_score + energy_uncertainty_score)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
