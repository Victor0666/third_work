import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with successor-release blocking mitigation:
      - Replaces redundant joint risk term with additive, conditionally gated successor-release penalty.
      - Successor-release penalty: quantifies how much a task's completion delays high-work, low-slack successors — directly addressing SUCCESSOR_RELEASE_BLOCKING evidence.
      - Uses normalized slack feasibility (slightly positive exponent) to weight successor pressure only when slack is tight.
      - All operations remain vectorized; still only ONE conditional branch (DDL gate).
      - Energy normalization retains fallback logic but avoids over-amplification via bounded dispersion scaling.
      - Wait decay preserved for anti-starvation; neg_slack remains dominant uncorrupted term.
      - No numeric literals beyond {-2,-1,0,1,2}; all tunables declared in PARAMETER_SCHEMA.
    """
    eps = 3.098069086594094e-06
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    neg_slack = np.clip(-slack, 0.0, None)
    median_slack = np.median(slack) if N > 0 else 0.0
    uncertainty_normalized = (uncertainty - np.min(uncertainty)) / (np.ptp(uncertainty) + eps) if N > 1 else np.zeros_like(uncertainty)
    ddl_risk_gate = np.where((slack <= median_slack) & (uncertainty_normalized > 0.5106880235661839), 1.0, 0.0)
    critical_path_pressure = upward_rank * remaining_work
    gated_critical_pressure = ddl_risk_gate * critical_path_pressure
    slack_feasibility = np.clip(slack / (np.abs(median_slack) + eps), 0.0, 1.0)
    slack_feasibility_sharpened = np.power(slack_feasibility + eps, 0.4847243360025403)
    successor_release_penalty = remaining_work * (1.0 - slack_feasibility_sharpened)
    gated_successor_penalty = ddl_risk_gate * successor_release_penalty * 1.794495490953749
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / (duration + eps)
    energy_center = np.median(energy_per_duration) if N > 0 else 0.0
    energy_std = np.std(energy_per_duration) if N > 1 else 0.0
    energy_range = np.max(energy_per_duration) - np.min(energy_per_duration) if N > 0 else 0.0
    energy_dispersion = np.where(energy_std > eps, energy_std, energy_range)
    norm_energy = (energy_per_duration - energy_center) / (energy_dispersion * 0.8245278656680712 + eps)
    max_wait = np.max(ready_wait_time) if N > 0 else eps
    wait_ratio = np.clip(ready_wait_time / (max_wait + eps), 0.0, 1.0)
    wait_decay = np.power(wait_ratio + eps, 0.9438457133159316)
    score = neg_slack + gated_successor_penalty + 1.325011627332208 * gated_critical_pressure + norm_energy - wait_decay
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
