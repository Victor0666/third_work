import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's robust median-MAD normalization and validated slack-pressure gates 
       with Parent 1's slack-aware uncertainty gating and dedicated starvation relief under pressure.
       Structural change: replaces fixed uncertainty_gate_threshold with adaptive variant scaled by sigmoid(slack),
       and unifies starvation relief into a single pressure-gated linear term — removing 'wait_saturation_offset'
       since logistic saturation was dropped in favor of bounded linear decay."""
    eps = 6.952254046894997e-05
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
    ddl_gate = 1.0 / (1.0 + np.exp(-4.141909335715953 * slack))
    adaptive_uncert_thresh = 0.14644694320980395 * (1.0 - 1.0 / (1.0 + np.exp(-slack)))
    slack_aware_uncert_gate = (norm_uncert > adaptive_uncert_thresh).astype(float) * ddl_gate
    ddl_breach = (slack <= 0.0).astype(float)
    critical_path_leverage = norm_rank * norm_work * ddl_breach
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 3.4332131360550666
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    coupled_slack = np.clip(norm_slack, -1.0, 1.0)
    rank_slack_coupling = 1.0 + 1.2600544057652407 * coupled_slack
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-4.68458904932655 * (slack_pressure - 1.0)))
    boosted_rank = norm_rank * rank_slack_coupling * (1.0 + 1.2600544057652407 * rank_gate)
    duration_risk_score = norm_duration * slack_aware_uncert_gate * slack_pressure
    wait_benefit_pressure = np.clip(0.37568683436776107 * ready_wait_time, 0.0, 2.0) * ddl_breach
    energy_uncert_penalty = norm_energy * norm_uncert * slack_aware_uncert_gate
    energy_slack_penalty = norm_energy * (1.0 + 3.4332131360550666 * slack_pressure) * ddl_gate
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(critical_path_leverage, -2.0, 2.0) - np.clip(boosted_rank, -2.0, 2.0) - 1.8845751775567252 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - np.clip(wait_benefit_pressure, -2.0, 2.0) + 0.9842983211261085 * np.clip(duration_risk_score, -2.0, 2.0) + 0.32557774520685034 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 3.4332131360550666 * np.clip(energy_slack_penalty, -2.0, 2.0) + 0.5877582938431036 * np.clip(norm_work, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
