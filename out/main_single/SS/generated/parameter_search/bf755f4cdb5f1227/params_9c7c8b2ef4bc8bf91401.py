import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule: replaces sigmoid gates with bounded linear interpolation;
       uses sign-preserving MAD normalization; removes inactive parameters (duration_robustness);
       introduces critical successor release interaction via upward_rank * remaining_work under ddl_gate;
       applies clipped linear DDL-protection gate instead of soft sigmoid for stronger hard-constraint enforcement;
       prioritizes tasks with smallest slack first, then critical path impact, then starvation relief."""
    eps = 2.1933174632693988e-05
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
            med = x[0]
            mad = eps
        else:
            med = np.median(x)
            mad = np.median(np.abs(x - med))
        spread = np.maximum(mad, eps)
        return (x - med) / spread
    norm_slack = mad_normalize(slack)
    norm_energy = mad_normalize(min_incremental_energy)
    norm_duration = mad_normalize(min_exec_time + min_comm_time)
    norm_rank = mad_normalize(upward_rank)
    norm_work = mad_normalize(remaining_work)
    norm_wait = mad_normalize(ready_wait_time)
    norm_uncert = mad_normalize(uncertainty)
    ddl_gate = 0.8509640759344679 * (1.0 + np.tanh(5.486531675387914 * slack))
    critical_path_leverage = norm_rank * norm_work * ddl_gate
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.729917442374453
    norm_slack_penalty = mad_normalize(raw_slack_penalty)
    pressure = np.clip(1.0 + 2.4454113935808732 * -norm_slack, 0.0, 2.0)
    uncert_gate = (norm_uncert >= 0.024163381858136615).astype(float)
    wait_benefit = np.clip(1.0 - np.exp(-0.18261842469805123 * ready_wait_time), 0.0, 0.7072063589609848)
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_gate
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(critical_path_leverage, -2.0, 2.0) - np.clip(norm_rank * pressure, -2.0, 2.0) - 1.3443847160193434 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + 0.18505857799150713 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 1.4563758297879188 * np.clip(norm_work, -2.0, 2.0) + 0.5921449340828391 * np.clip(norm_rank * norm_work * (1.0 - ddl_gate), -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
