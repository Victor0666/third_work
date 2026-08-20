import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: replaces deprecated duration_robustness with active comm-energy coupling;
       introduces successor-release gate to prioritize unblocking of critical downstream work;
       removes inactive wait_saturation_offset and uses robust logistic starvation relief;
       retains median-MAD normalization and strict DDL-first semantics via dual-gated terms."""
    eps = 0.0006491772842550722
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
    norm_comm_energy = median_mad_normalize(min_comm_time * min_incremental_energy)
    ddl_gate = 1.0 / (1.0 + np.exp(-5.2942095420507655 * slack))
    ddl_breach = (slack <= 0.0).astype(float)
    critical_path_leverage = norm_rank * norm_work * ddl_breach
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.9812234749749769
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-1.7443254065818632 * (slack_pressure - 1.0)))
    boosted_rank = norm_rank * (1.0 + 2.0957175494066274 * rank_gate)
    release_horizon = remaining_work / (upward_rank + eps)
    successor_release_gate = 1.0 / (1.0 + np.exp(-6.417065162531255 * (slack - release_horizon)))
    wait_benefit = 1.0 - np.exp(-0.35323003214949167 * ready_wait_time)
    uncert_gate = 1.0 / (1.0 + np.exp(-5.2942095420507655 * (norm_uncert - 0.7121209340067354)))
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_gate
    comm_energy_penalty = norm_comm_energy * ddl_gate * slack_pressure
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(critical_path_leverage, -2.0, 2.0) - np.clip(boosted_rank, -2.0, 2.0) - 0.106356664815911 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + 1.4095928929064183 * np.clip(comm_energy_penalty, -2.0, 2.0) + 0.9121306824084937 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 0.16127281376670444 * np.clip(norm_work, -2.0, 2.0) - np.clip(successor_release_gate, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
