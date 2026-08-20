import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's robust median-MAD normalization and sharpened slack gates 
       with Parent 1's validated critical-path coupling and slack-aware uncertainty gating.
       Key improvements:
         - Replaces fixed uncertainty_gate_threshold with adaptive version: threshold rises as slack approaches zero
         - Introduces dedicated starvation relief *only* under deadline pressure (slack <= 0), decoupled from congestion
         - Combines both critical-path signals: (1) boosted_rank (Parent 2) + (2) critical_coupling_under_pressure (Parent 1)
         - Uses clipped MAD-normalized slack for both penalty and gating to enhance sensitivity near deadlines
         - All terms bounded via [-2,2] clipping; final score deterministic, finite, shape-(N,)."""
    eps = 1.85271580482046e-05
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
    adaptive_uncert_thresh = 0.32056224118732735 * (1.0 - 1.0 / (1.0 + np.exp(-2.0 * slack)))
    uncert_gate_adaptive = (norm_uncert > adaptive_uncert_thresh).astype(float) * (slack >= 0.0).astype(float)
    ddl_gate = 1.0 / (1.0 + np.exp(-7.532901936048398 * slack))
    ddl_breach = (slack <= 0.0).astype(float)
    critical_coupling = norm_rank * norm_work * ddl_breach * 2.3488479511384304
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 2.490987790088353
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    coupled_slack = np.clip(norm_slack, -1.0, 1.0)
    rank_slack_coupling = 1.0 + 2.3488479511384304 * coupled_slack
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-3.21586895839144 * (slack_pressure - 1.0)))
    boosted_rank = norm_rank * rank_slack_coupling * (1.0 + 2.3488479511384304 * rank_gate)
    duration_risk_score = norm_duration * uncert_gate_adaptive * slack_pressure * ddl_gate
    wait_benefit_pressure = (1.0 - np.exp(-0.7345920836028987 * (ready_wait_time + 2.330415199561957e-06))) * ddl_breach
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate_adaptive * ddl_gate
    energy_slack_penalty = norm_energy * (1.0 + 2.490987790088353 * slack_pressure) * ddl_gate
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(critical_coupling, -2.0, 2.0) - np.clip(boosted_rank, -2.0, 2.0) - 0.42270872659960723 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - np.clip(wait_benefit_pressure, -2.0, 2.0) + 1.8070534711463693 * np.clip(duration_risk_score, -2.0, 2.0) + 0.4017925861111905 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 2.490987790088353 * np.clip(energy_slack_penalty, -2.0, 2.0) + 0.7704582124104672 * np.clip(norm_work, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
