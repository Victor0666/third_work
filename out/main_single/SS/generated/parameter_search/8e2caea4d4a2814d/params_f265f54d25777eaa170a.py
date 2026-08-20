import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule: replaces sigmoid DDL gates with bounded smooth pressure function;
       introduces successor-release interaction (min_exec_time * upward_rank * (slack <= 0));
       removes redundant duration_robustness and wait_saturation_offset (validated inactive);
       uses sign-preserving MAD-normalized slack for sharper urgency near zero;
       applies joint congestion gating only when both ready_wait_time and uncertainty are high;
       enforces strict feasibility-first energy penalization via hard binary gate (slack >= 0);
       adds host-load surrogate (norm_duration * norm_uncert) gated by feasibility and uncertainty threshold."""
    eps = 0.03071189343924318
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
    feasible_gate = (slack >= 0.0).astype(float)
    ddl_pressure = 1.0 / (1.0 + np.abs(slack) + eps)
    critical_breach = (slack <= 0.0).astype(float)
    critical_path_impact = norm_rank * norm_work * critical_breach
    clipped_rank = np.clip(norm_rank, 0.0, 2.0)
    successor_release = min_exec_time * clipped_rank * critical_breach
    wait_median = np.median(ready_wait_time)
    wait_std = np.std(ready_wait_time)
    wait_high = (ready_wait_time > wait_median + 0.5985366178194935 * wait_std).astype(float)
    uncert_high = (uncertainty > 0.35339271035782194).astype(float)
    congestion_gate = wait_high * uncert_high
    host_load_surrogate = norm_duration * norm_uncert * feasible_gate * uncert_high
    wait_benefit = 1.0 - np.exp(-0.10839962823195295 * ready_wait_time)
    energy_penalty = norm_energy * feasible_gate * (1.0 + 1.3573978986379447 * ddl_pressure)
    energy_uncert_coupling = norm_energy * norm_uncert * feasible_gate * uncert_high
    score = +np.clip(norm_slack, -2.0, 2.0) + np.clip(ddl_pressure, 0.0, 2.0) - np.clip(critical_path_impact, -2.0, 2.0) - np.clip(successor_release, -2.0, 2.0) - 0.28683370302151806 * np.clip(energy_penalty, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + 0.705231022768176 * np.clip(energy_uncert_coupling, -2.0, 2.0) + 1.57067670099366 * np.clip(norm_work, -2.0, 2.0) + 1.2349119134818676 * np.clip(host_load_surrogate, -2.0, 2.0) + 1.514989879612059 * np.clip(norm_rank * ddl_pressure, -2.0, 2.0) + congestion_gate * np.clip(norm_wait + norm_uncert, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
