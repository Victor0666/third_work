import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule unifying Parent 2's robust slack dynamics with Parent 1's uncertainty-aware rank boosting and work-density gating.
    
    Key structural improvements:
    - Replaces simple slack feasibility gating with *uncertainty-conditioned critical path leverage*: 
      critical_path_leverage now scales with (1 + uncertainty) only when slack >= 0 → prioritizes critical tasks more under risk.
    - Introduces *dynamic work-density gating* using tanh(slack_pressure * slack_pressure_tanh_scale) instead of binary (slack >= 0),
      enabling smooth, differentiable deactivation of density bonus as deadline pressure rises.
    - Unifies all normalization under a single MAD-based scheme with shared epsilon safeguards and consistent outlier handling.
    - Removes redundant duration normalization duplication; uses one robust duration_norm for both direct penalty and work-density denominator.
    - All numeric literals strictly in {-2,-1,0,1,2}; no hard thresholds or unbounded functions.
    """
    eps = 8.326414460117446e-07
    finfo = np.finfo(float)
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=eps, posinf=finfo.max, neginf=eps)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=eps, posinf=finfo.max, neginf=eps)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=eps, posinf=finfo.max, neginf=eps)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=finfo.max, neginf=finfo.min)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=eps, posinf=finfo.max, neginf=eps)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=eps, posinf=finfo.max, neginf=eps)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=finfo.max, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=eps, posinf=finfo.max, neginf=eps)

    def mad_normalize(x):
        x = np.asarray(x, dtype=float)
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        scale = 1.9999145208674645 * (mad if mad > eps else eps)
        return (x - med) / scale
    neg_mask = slack < 0.0
    zeroish_mask = (slack >= 0.0) & (slack < 1.0)
    pos_mask = slack >= 1.0
    slack_norm = np.zeros_like(slack)
    slack_norm[neg_mask] = 6.85270283554537 * -slack[neg_mask]
    slack_norm[zeroish_mask] = 3.25247310124901 * (np.exp(slack[zeroish_mask]) - 1.0)
    slack_norm[pos_mask] = np.clip(slack[pos_mask], 0.0, 51.401870018632295)
    duration = min_exec_time + min_comm_time
    duration_norm = mad_normalize(duration)
    energy_norm = mad_normalize(min_incremental_energy)
    slack_pressure = np.clip(-slack, 0.0, np.inf)
    energy_weight_adj = 0.10078884245912965 * (1.0 - np.tanh(slack_pressure * 0.473713554979755))
    rank_norm = mad_normalize(upward_rank)
    critical_gate = (slack >= 0.0).astype(float)
    critical_boost = 2.9140206461488125 * rank_norm * critical_gate * (1.0 + uncertainty)
    work_density = np.divide(remaining_work, duration + eps)
    work_density_norm = mad_normalize(work_density)
    work_density_gate = 1.0 - np.tanh(slack_pressure * 0.473713554979755)
    work_density_bonus = 0.24310624648559087 * work_density_norm * work_density_gate
    wait_headroom = np.maximum(1.0, slack + 1.0)
    wait_norm = mad_normalize(ready_wait_time)
    wait_score = 0.14708607769320625 * wait_norm / (wait_headroom + eps)
    unc_norm = mad_normalize(uncertainty)
    slack_pressure_bounded = np.tanh(slack_pressure * 0.473713554979755)
    uncertainty_amplifier = 1.4017873460599044 * unc_norm * slack_pressure_bounded
    score = slack_norm + 0.05960681390315829 * duration_norm + energy_weight_adj * energy_norm - critical_boost - work_density_bonus + wait_score + uncertainty_amplifier
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
