import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule: merges Parent 2's numerical stability with Parent 1's structured slack-pressure gating;
       introduces novel sigmoid-based slack-pressure gate for critical-path coupling — smoother and more discriminative than linear ramp;
       retains Parent 2's strict DDL-protection gate to disable all risk-sensitive penalties when slack < 0;
       replaces raw duration penalty with robustified duration-uncertainty product under DDL pressure;
       adds explicit successor-release weight parameter for calibrated downstream impact;
       uses median-MAD normalization throughout for outlier resilience and sign preservation."""
    eps = 7.242028182217588e-05
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
    ddl_gate = np.clip(slack, 0.0, 1.0)
    ddl_gate = np.where(slack < 0.0, 0.0, ddl_gate)
    slack_pressure = 1.0 / (1.0 + np.exp(-1.6832826239479142 * -norm_slack))
    successor_release = 0.45942128253421527 * norm_rank * norm_work * slack_pressure
    coupled_rank = norm_rank * (1.0 + 1.0334897245210644 * slack_pressure)
    uncert_gate = np.where(norm_uncert > 0.007186594602082273, 1.0, 0.0)
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_gate
    wait_benefit = np.clip(0.14871495688924868 * ready_wait_time, 0.0, 1.0)
    duration_uncert_penalty = norm_duration * norm_uncert * slack_pressure * ddl_gate
    raw_slack_penalty = np.maximum(-slack, 0.0)
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(successor_release, -2.0, 2.0) - np.clip(coupled_rank, -2.0, 2.0) - 0.26705710475468303 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip(duration_uncert_penalty, -2.0, 2.0) + 0.5367142441529058 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 0.42093710159854925 * np.clip(norm_work, -2.0, 2.0) + np.clip(slack_pressure * 6.061704111295361, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
