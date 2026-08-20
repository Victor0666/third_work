import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule: replaces sigmoid gates with bounded linear interpolation for monotonic stability;
       removes redundant duration_robustness and wait_saturation_offset (evidence shows near-zero variance);
       introduces successor-release coupling via norm_work * norm_rank interaction, gated by slack pressure;
       uses clipped linear slack mapping instead of power-law for better numerical behavior under tight DDL;
       applies median-MAD normalization per feature with N-aware fallback; enforces deterministic finite output."""
    eps = 0.05678492252051663
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
    norm_rank = median_mad_normalize(upward_rank)
    norm_work = median_mad_normalize(remaining_work)
    norm_wait = median_mad_normalize(ready_wait_time)
    norm_uncert = median_mad_normalize(uncertainty)
    ddl_feasible = (slack >= 0.0).astype(float)
    slack_pressure = np.clip(norm_slack, -1.0, 1.0)
    pressure_gate = 0.6018058251187012 * (1.0 + slack_pressure)
    successor_coupling = norm_work * norm_rank * pressure_gate
    critical_boost = norm_rank * (1.0 + 1.6175084712869696 * pressure_gate)
    wait_benefit = 1.0 - np.exp(-0.7190226533928323 * ready_wait_time)
    wait_benefit = np.clip(wait_benefit, 0.0, 0.9878446773519237)
    uncert_gate = (norm_uncert > 0.12195925187743571).astype(float)
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_feasible
    energy_slack_sensitivity = 1.2034039612906509 * (1.0 + 2.8741581029074745 * (1.0 - pressure_gate))
    score = +np.clip(-norm_slack, -2.0, 2.0) - np.clip(critical_boost, -2.0, 2.0) - np.clip(successor_coupling, -2.0, 2.0) - energy_slack_sensitivity * np.clip(norm_energy * ddl_feasible, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + 0.6031446374118952 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 0.4964020428444957 * np.clip(norm_work, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
