import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: replaces unstable sigmoid host-load gating with bounded linear ramp (robust, monotonic);
       restores signed MAD-normalized slack as primary urgency signal — avoids noise amplification from exponentiation;
       reinstates unconditional starvation relief (no slack sign gating) to prevent near-feasible starvation;
       retains robust ddl_pressure, successor-release, and critical-path urgency from prior versions;
       all feature interactions clipped to [-2,2] to enforce bounded AST depth and deterministic stability."""
    eps = 0.002106775259241349
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
    ddl_pressure = 1.0 / (1.0 + np.abs(slack) + eps)
    ddl_breach = (slack <= 0.0).astype(float)
    critical_path_urgency = norm_rank * norm_work * ddl_breach
    successor_release = min_exec_time * norm_rank * ddl_breach
    norm_slack_penalty = np.clip(norm_slack, -2.0, 2.0)
    load_activation = norm_duration * norm_uncert
    host_load_ramp = np.clip(1.903397977139176 * (load_activation - 0.4876854969857931), 0.0, 1.0)
    host_load_surrogate = load_activation * host_load_ramp * ddl_pressure
    energy_uncert_gate = (ddl_pressure > 0.9130756054848285).astype(float) * (norm_uncert >= 0.5460653332857619).astype(float)
    energy_uncert_penalty = norm_energy * norm_uncert * energy_uncert_gate
    wait_benefit = np.exp(-0.03561798106643104 * ready_wait_time) * (1.0 - ddl_pressure)
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(critical_path_urgency, -2.0, 2.0) - np.clip(successor_release, -2.0, 2.0) - 0.2510797914863358 * np.clip(norm_energy * ddl_pressure, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip(host_load_surrogate, -2.0, 2.0) + 0.5429967327462556 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 1.0430501602651083 * np.clip(norm_work, -2.0, 2.0) + 1.0344987329705542 * np.clip(norm_rank * ddl_pressure, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
