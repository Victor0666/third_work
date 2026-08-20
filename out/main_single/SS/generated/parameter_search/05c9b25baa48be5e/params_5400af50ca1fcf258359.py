import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule: replaces sigmoid DDL gates with bounded smooth pressure function;
       introduces successor-release interaction via min_exec_time × upward_rank × (slack <= 0);
       replaces median-MAD with robust MAD-only scaling (no centering) to preserve absolute urgency;
       removes redundant duration_robustness and wait_saturation_offset per diagnostics;
       enforces strict feasibility-first via hard binary ddl_gate for energy/uncertainty terms;
       adds host-load surrogate: norm_duration × norm_uncert gated by both slack >= 0 and uncertainty > threshold."""
    eps = 8.819151859235085e-06
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def mad_normalize(x):
        x = np.copy(x)
        if N == 1:
            mad = eps
        else:
            mad = np.median(np.abs(x - np.median(x)))
        spread = mad if mad > eps else eps
        return x / spread
    norm_slack = mad_normalize(slack)
    norm_energy = mad_normalize(min_incremental_energy)
    norm_duration = mad_normalize(min_exec_time + min_comm_time)
    norm_rank = mad_normalize(upward_rank)
    norm_work = mad_normalize(remaining_work)
    norm_wait = mad_normalize(ready_wait_time)
    norm_uncert = mad_normalize(uncertainty)
    ddl_gate = (slack >= 0.0).astype(float)
    ddl_breach = (slack < 0.0).astype(float)
    critical_path_leverage = norm_rank * norm_work * ddl_breach
    successor_release = min_exec_time * norm_rank * ddl_breach
    norm_successor_release = mad_normalize(successor_release)
    ddl_pressure = 1.0 / (1.0 + np.abs(slack) + eps)
    load_surrogate_gate = (uncertainty > 0.6964413222380637).astype(float) * ddl_gate
    host_load_surrogate = norm_duration * norm_uncert * load_surrogate_gate
    wait_benefit = 1.0 - np.exp(-0.11133309477238232 * ready_wait_time)
    wait_benefit = np.clip(wait_benefit, 0.0, 1.0 - eps)
    energy_uncert_gate = load_surrogate_gate
    energy_uncert_penalty = norm_energy * norm_uncert * energy_uncert_gate
    score = +np.clip(norm_slack, -2.0, 2.0) - np.clip(critical_path_leverage, -2.0, 2.0) - np.clip(norm_successor_release, -2.0, 2.0) - 0.6491442697337018 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + 1.2410485975192502 * np.clip(norm_work, -2.0, 2.0) + 6.633764869786905 * np.clip(host_load_surrogate, -2.0, 2.0) + 0.47372177827931766 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 2.4615621350967327 * np.clip(ddl_pressure, -2.0, 2.0) + 0.5142478958303782 * np.clip(norm_rank * ddl_pressure, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
