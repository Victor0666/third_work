import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: restores logistic starvation relief (using wait_decay + epsilon offset);
       introduces host-load–conditional gate using (min_exec_time + min_comm_time) as VM load proxy;
       reverts to static uncertainty_gate_threshold per diagnostic evidence;
       retains robust median-MAD normalization and all validated DDL-protection mechanisms.
       Key structural change: host-load gate multiplicatively modulates energy_sensitivity to avoid over-prioritizing low-energy tasks on overloaded VMs."""
    eps = 6.515580280252523e-05
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
    ddl_gate = 1.0 / (1.0 + np.exp(-3.8731873891833026 * slack))
    host_load_gate = 1.0 / (1.0 + np.exp(-7.237928579852103 * norm_duration))
    ddl_breach = (slack <= 0.0).astype(float)
    critical_path_leverage = norm_rank * norm_work * ddl_breach
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 3.5924853172242175
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    coupled_slack = np.clip(norm_slack, -1.0, 1.0)
    rank_slack_coupling = 1.0 + 1.7362412178929996 * coupled_slack
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-7.922056869584017 * (slack_pressure - 1.0)))
    boosted_rank = norm_rank * rank_slack_coupling * (1.0 + 1.7362412178929996 * rank_gate)
    uncert_gate = (norm_uncert > 0.27502534133656914).astype(float) * ddl_gate
    duration_risk_score = norm_duration * uncert_gate * slack_pressure
    wait_benefit = 1.0 - np.exp(-0.46541347505172476 * (ready_wait_time + eps))
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_gate
    energy_slack_penalty = norm_energy * (1.0 + 3.5924853172242175 * slack_pressure) * ddl_gate
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(critical_path_leverage, -2.0, 2.0) - np.clip(boosted_rank, -2.0, 2.0) - 1.387784276092272 * host_load_gate * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + 0.959411439145744 * np.clip(duration_risk_score, -2.0, 2.0) + 0.615206723454117 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 3.5924853172242175 * np.clip(energy_slack_penalty, -2.0, 2.0) + 0.8569708240517385 * np.clip(norm_work, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
