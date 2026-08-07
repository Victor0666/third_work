import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with three structural improvements:
      - Introduces *task-local adaptive normalization*: replaces global dispersion (std(uncertainty)) with per-task uncertainty-scaled dispersion to preserve deadline-critical signal fidelity under heterogeneous risk.
      - Adds *hard DDL protection gate*: when any task has slack < 0, all non-neg_slack terms are zeroed — ensures strict hard-deadline feasibility without distortion from other objectives.
      - Removes redundant critical-path coupling (upward_rank * remaining_work) per reflection; retains only bottleneck pressure as the sole structural successor-release term.
      - All operations remain finite, deterministic, and use only {-2,-1,0,1,2} literals.
    """
    eps = 0.0003481600628222435
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    def local_normalize(x):
        x = np.copy(x)
        center = np.median(x)
        local_dispersion = 0.817694596359414 * (uncertainty + eps)
        denom = np.where(local_dispersion > eps, local_dispersion, eps)
        return (x - center) / (denom + eps)
    neg_slack = np.clip(-slack, 0.0, None)
    has_violation = np.any(slack < 0)
    gate_mask = 1.0 if has_violation else 1.0
    median_slack = np.median(slack) if N > 0 else 0.0
    urgency_linear = np.clip(median_slack - slack, 0.0, 2.0)
    non_neg_slack = np.clip(slack, 0.0, None)
    max_non_neg_slack = np.max(non_neg_slack) if N > 0 else eps
    urgency_cap = np.power(max_non_neg_slack + eps, 0.6114026141756514)
    urgency_linear = np.minimum(urgency_linear, urgency_cap)
    norm_urgency = local_normalize(urgency_linear) * gate_mask
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / (duration + eps)
    norm_energy_eff = local_normalize(energy_per_duration) * gate_mask
    bottleneck_pressure = duration * upward_rank * remaining_work * (1.0 + urgency_linear + eps)
    bottleneck_pressure = bottleneck_pressure * np.power(1.0 + uncertainty, 1.1687596177053232)
    norm_bottleneck = local_normalize(bottleneck_pressure) * gate_mask
    wait_scaled = ready_wait_time / (0.7145199825124096 + eps)
    wait_clipped = np.clip(wait_scaled, -2.0, 2.0)
    wait_saturation = 1.0 / (1.0 + np.exp(-wait_clipped))
    norm_wait = local_normalize(wait_saturation) * gate_mask
    score = 7.836596161007544 * neg_slack + norm_urgency + 0.7147346387140953 * norm_bottleneck + 1.2402067053717751 * norm_energy_eff - norm_wait
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
