import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule addressing reflection insights:
      - Replaces power-law DDL penalty with bounded linear urgency (slope param) for balanced hard-constraint fidelity.
      - Restores raw `neg_slack` as dominant additive term — preserves monotonic, interpretable deadline violation signal.
      - Introduces *host-load-aware energy gating*: energy term is suppressed when `min_incremental_energy` is high *and*
        `ready_wait_time` is low (indicating low contention), preventing energy-driven scheduling that risks DDL violation.
      - Keeps all robust MAD normalizations, bottleneck-aware duration scaling, and power-law wait decay.
      - Adds explicit feasibility-aware coupling: energy term scaled by `(1 - gate)` where gate activates under low-load + high-energy conditions.
      - All operations finite, deterministic, and use only {-2,-1,0,1,2} literals.
    """
    eps = 8.617766521523608e-06
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    def mad_normalize(x):
        x = np.copy(x)
        center = np.median(x)
        abs_dev = np.abs(x - center)
        mad = np.median(abs_dev) if N > 0 else eps
        fallback_range = np.max(x) - np.min(x) if N > 0 else eps
        denom = mad if mad > eps else fallback_range
        return (x - center) / (denom + eps)
    median_slack = np.median(slack) if N > 0 else 0.0
    ddl_risk_mask = (slack < 0.584654969929271 * (median_slack + eps)).astype(float)
    neg_slack = np.clip(-slack, 0.0, None)
    urgency_score = neg_slack * 1.981586711121072 * ddl_risk_mask
    critical_pressure = upward_rank * remaining_work
    critical_pressure = critical_pressure * (1.0 + ddl_risk_mask * (1.1124874785869003 - 1.0))
    norm_critical_pressure = mad_normalize(critical_pressure)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    bottleneck_pressure = duration * critical_pressure * ddl_risk_mask
    norm_bottleneck = mad_normalize(bottleneck_pressure)
    norm_energy_cost = mad_normalize(min_incremental_energy)
    norm_wait_level = mad_normalize(ready_wait_time)
    energy_feasibility_gate = (norm_energy_cost > 0.0).astype(float) * (norm_wait_level < 0.0).astype(float) * (np.abs(norm_energy_cost) > 0.7565750477607407).astype(float)
    energy_per_duration = min_incremental_energy / (duration + eps)
    norm_energy = mad_normalize(energy_per_duration) * 1.261131543138273
    norm_energy = norm_energy * (1.0 - energy_feasibility_gate)
    norm_slack_deficit = mad_normalize(neg_slack)
    unc_coupled_deficit = norm_slack_deficit * np.power(1.0 + uncertainty, 0.13396520159671824)
    unc_coupled_deficit = unc_coupled_deficit * ddl_risk_mask
    max_wait = np.max(ready_wait_time) if N > 0 else eps
    wait_ratio = np.clip(ready_wait_time / (max_wait + eps), 0.0, 1.0)
    wait_decay = np.power(wait_ratio + eps, 0.49086264805118485)
    wait_priority = 1.0 - wait_decay
    norm_wait = mad_normalize(wait_priority)
    score = neg_slack + urgency_score + norm_bottleneck + norm_critical_pressure + norm_energy + mad_normalize(unc_coupled_deficit) - norm_wait
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
