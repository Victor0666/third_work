import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule: hybridizes Parent 2's stability with Parent 1's safe-slack gating.
       Novel structural change: dual-gate energy suppression — (i) hard DDL-protection gate (slack < 0 → disable), 
       and (ii) soft safe-slack gate (norm_slack > threshold → suppress), enabling robust feasibility-first behavior
       across all slack regimes. Introduces sigmoid-coupled successor-release interaction for sharper critical-path activation.
       Replaces linear ddl_gate with clipped sigmoid on normalized slack to improve gradient continuity while avoiding overflow."""
    eps = 3.0239809131175003e-06
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def median_mad_normalize(x):
        x = np.copy(x)
        if N == 1:
            med = x[0]
            mad = eps
        else:
            med = np.median(x)
            mad = np.median(np.abs(x - med))
        spread = mad if mad > eps else eps
        return (x - med) / spread
    norm_slack = median_mad_normalize(slack)
    norm_energy = median_mad_normalize(min_incremental_energy)
    norm_duration = median_mad_normalize(min_exec_time + min_comm_time)
    norm_rank = median_mad_normalize(upward_rank)
    norm_work = median_mad_normalize(remaining_work)
    norm_wait = median_mad_normalize(ready_wait_time)
    norm_uncert = median_mad_normalize(uncertainty)
    hard_ddl_gate = np.where(slack < 0.0, 0.0, 1.0)
    safe_slack_mask = norm_slack > 0.25234115027917026
    soft_energy_suppress = np.where(safe_slack_mask, 0.0, 1.0)
    energy_enabled = hard_ddl_gate * soft_energy_suppress
    slack_pressure_norm = np.clip(-norm_slack, 0.0, 2.0)
    slack_pressure = 1.0 / (1.0 + np.exp(-4.738291346510869 * (slack_pressure_norm - 1.0)))
    successor_release = norm_rank * norm_work * slack_pressure
    raw_slack_penalty = np.maximum(-slack, 0.0)
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    coupled_rank = norm_rank * (1.0 + 1.4825441631040963 * slack_pressure)
    uncert_gate = np.where(norm_uncert > 0.24783947798622585, 1.0, 0.0)
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * energy_enabled
    wait_benefit = np.clip(0.09514258742246706 * ready_wait_time, 0.0, 1.0)
    exec_penalty = norm_duration * slack_pressure * hard_ddl_gate
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(successor_release, -2.0, 2.0) - np.clip(coupled_rank, -2.0, 2.0) - 0.5490524813064751 * np.clip(norm_energy * energy_enabled, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip(exec_penalty, -2.0, 2.0) + 0.7065307749689551 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 0.3303992684232292 * np.clip(norm_work, -2.0, 2.0) + np.clip(slack_pressure * 3.9952577132971157, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
