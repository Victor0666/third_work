import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: simplifies gates to eliminate instability from dual-mode slack pressure;
       restores and isolates the empirically validated 'critical_path_leverage' as dominant DDL-breach signal;
       replaces ad-hoc starvation gating with a monotonic, slack-distance-aware linear ramp;
       uses robust median-MAD normalization throughout; all terms clipped to [-2,2] for stability;
       removes inactive parameters (duration_robustness, wait_saturation_offset, energy_uncertainty_interaction)
       and redundant interactions to reduce noise and improve interpretability."""
    eps = 3.794761304542853e-05
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
    ddl_gate = 1.0 / (1.0 + np.exp(-3.265048809626908 * slack))
    ddl_breach = (slack <= 0.0).astype(float)
    critical_path_leverage = norm_rank * norm_work * ddl_breach
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 2.4579829352708162
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    coupled_slack = np.clip(norm_slack, -1.0, 1.0)
    rank_slack_coupling = 1.0 + 1.8209772562428448 * coupled_slack
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-3.268385087725673 * (slack_pressure - 1.0)))
    boosted_rank = norm_rank * rank_slack_coupling * (1.0 + 1.8209772562428448 * rank_gate)
    uncert_gate = 1.0 / (1.0 + np.exp(-3.265048809626908 * (norm_uncert - 0.5377551907449404)))
    slack_distance = np.clip(slack, 0.0, np.inf)
    norm_slack_distance = median_mad_normalize(slack_distance)
    starvation_relief = np.clip(norm_slack_distance, 0.0, 1.0)
    energy_preference = norm_energy * ddl_gate * (1.0 - uncert_gate)
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - 1.0276400861973538 * np.clip(critical_path_leverage, -2.0, 2.0) - np.clip(boosted_rank, -2.0, 2.0) - 0.9762788965857255 * np.clip(energy_preference, -2.0, 2.0) + 0.20679200856952448 * np.clip(starvation_relief, -2.0, 2.0) + 0.5930957569875605 * np.clip(norm_work, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
