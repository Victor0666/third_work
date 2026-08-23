import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved hybrid priority rule combining Parent 2's stability with Parent 1's joint DDL-uncertainty gating.
    
    Key structural improvements:
    - Replaces piecewise slack with smooth, bounded tanh-based urgency curve (improved cross-seed robustness).
    - Introduces *joint DDL-uncertainty gate*: (slack >= 0) & (uncertainty <= median_uncertainty + gate_width)
      for critical_path_leverage and work_density_bonus — enforces dual safety before rewarding importance/density.
    - Uses linear fairness: wait_norm / (|slack| + 1), avoiding exponential overboosting of stale tasks under deadline pressure.
    - All features normalized via MAD with Gaussian-consistent scaling; all divisions epsilon-guarded.
    - No unbounded operations, no hidden state, fully deterministic.
    """
    eps = 0.09999754706122542
    finfo = np.finfo(float)
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=finfo.max, neginf=finfo.min)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=finfo.max, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=eps, posinf=finfo.max, neginf=eps)

    def mad_normalize(x):
        x = np.asarray(x, dtype=float)
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        scale = 1.4178329028229193 * (mad if mad > eps else eps)
        return (x - med) / scale
    slack_pressure = np.tanh(-slack * 0.1989199004471821)
    slack_norm = 1.5728587830559044 * slack_pressure + 0.9552656328033322 * (1.0 - np.tanh(slack * 0.1989199004471821))
    duration = min_exec_time + min_comm_time
    duration_norm = mad_normalize(duration)
    energy_norm = mad_normalize(min_incremental_energy)
    energy_weight_adj = 0.8998106334173663 * (1.0 - np.tanh(np.abs(slack) * 0.1989199004471821))
    rank_norm = mad_normalize(upward_rank)
    median_uncertainty = np.median(uncertainty)
    ddl_uncertainty_gate = ((slack >= 0.0) & (uncertainty <= median_uncertainty + 0.49789577197025786)).astype(float)
    critical_boost = 0.005865772778025876 * rank_norm * ddl_uncertainty_gate
    work_density = np.divide(remaining_work, duration + eps)
    work_density_norm = mad_normalize(work_density)
    work_density_bonus = 0.08369183783193607 * work_density_norm * ddl_uncertainty_gate
    wait_norm = mad_normalize(ready_wait_time)
    wait_score = 0.9493299023446473 * wait_norm / (np.abs(slack) + 1.0)
    unc_norm = mad_normalize(uncertainty)
    slack_pressure_bounded = np.tanh(np.abs(slack) * 0.1989199004471821)
    uncertainty_amplifier = 0.6772099366855762 * unc_norm * slack_pressure_bounded
    score = slack_norm + 0.2512952057477528 * duration_norm + energy_weight_adj * energy_norm - critical_boost - work_density_bonus + wait_score + uncertainty_amplifier
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
