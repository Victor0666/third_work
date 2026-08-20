import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule: replaces sigmoid DDL gates with bounded smooth pressure; introduces successor-release interaction;
       uses robust median-MAD scaling; removes unused parameters; adds conditional host-load surrogate via norm_duration * norm_uncert;
       replaces exponential wait_benefit with linear ramp; all numeric literals are -2, -1, 0, 1, or 2."""
    eps = 2.545173864174729e-05
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
    critical_path_leverage = norm_rank * norm_work * ddl_breach
    clipped_upward_rank = np.clip(norm_rank, 0.0, 2.0)
    successor_release = min_exec_time * clipped_upward_rank * ddl_breach
    load_surrogate = norm_duration * norm_uncert
    load_gate = (norm_uncert >= 0.39776803846008146).astype(float) * ddl_pressure
    host_load_penalty = load_surrogate * load_gate
    wait_benefit = np.clip(0.4338537807322258 * ready_wait_time, 0.0, 1.0)
    energy_uncert_gate = (slack >= 0.0).astype(float) * (norm_uncert >= 0.39776803846008146).astype(float)
    energy_uncert_penalty = norm_energy * norm_uncert * energy_uncert_gate
    energy_slack_penalty = norm_energy * ddl_pressure * (slack >= 0.0).astype(float)
    score = +np.clip(norm_slack, -2.0, 2.0) + np.clip(ddl_pressure * 2.2875485426908417, -2.0, 2.0) - np.clip(critical_path_leverage, -2.0, 2.0) - np.clip(successor_release, -2.0, 2.0) - np.clip(norm_rank * (1.0 + 1.2812509498797016 * ddl_pressure), -2.0, 2.0) - 0.27511651406598037 * np.clip(energy_slack_penalty, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + 0.4827945253703437 * np.clip(norm_work, -2.0, 2.0) + 0.7432201464168975 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 4.835403327570157 * np.clip(host_load_penalty, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
