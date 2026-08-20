import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: replaces duration_robustness with host-load-aware marginal-energy gating;
       eliminates wait_saturation_offset and uses bounded linear starvation relief conditioned on slack;
       introduces additive critical-path release priority (not multiplicative) gated by breach AND successor criticality;
       preserves median-MAD normalization, DDL-protection gate, and all stability safeguards."""
    eps = 0.00023717300463347224
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
    ddl_gate = 1.0 / (1.0 + np.exp(-3.1810018804878712 * slack))
    ddl_breach = (slack <= 0.0).astype(float)
    uncert_gate = 1.0 / (1.0 + np.exp(-3.1810018804878712 * (norm_uncert - 0.6981251304757312)))
    host_load_gate = 1.0 / (1.0 + np.exp(-3.1810018804878712 * (norm_duration - 0.989634617013648)))
    load_pressure = np.clip(norm_duration, 0.0, 2.0)
    marginal_energy_penalty = norm_energy * host_load_gate * load_pressure * ddl_gate
    successor_critical = (upward_rank > np.median(upward_rank)).astype(float)
    critical_path_release = 1.9638806248394876 * norm_rank * ddl_breach * successor_critical
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.0090114610748135
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    coupled_slack = np.clip(norm_slack, -1.0, 1.0)
    rank_slack_coupling = 1.0 + 2.2519170960010095 * coupled_slack
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-3.883874581305305 * (slack_pressure - 1.0)))
    boosted_rank = norm_rank * rank_slack_coupling * (1.0 + 2.2519170960010095 * rank_gate)
    wait_benefit = np.clip(0.5082176951087518 * ready_wait_time, 0.0, 1.0) * (1.0 - ddl_gate)
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_gate
    energy_slack_penalty = norm_energy * (1.0 + 1.0090114610748135 * slack_pressure) * ddl_gate
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) + critical_path_release - np.clip(boosted_rank, -2.0, 2.0) - 0.10596106767405028 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + 0.7042729573463297 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 1.0090114610748135 * np.clip(energy_slack_penalty, -2.0, 2.0) + 7.632082371849116e-05 * np.clip(norm_work, -2.0, 2.0) + np.clip(marginal_energy_penalty, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
