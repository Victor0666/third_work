import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule: merges Parent 2's numerical stability with Parent 1's structured slack-pressure gating;
       introduces novel sigmoid-based slack-pressure gate for critical-path coupling — smoother and more discriminative than linear ramp;
       retains Parent 2's strict DDL-protection gate to disable all risk-sensitive penalties when slack < 0;
       replaces raw duration penalty with robustified duration-uncertainty product under DDL pressure;
       adds explicit successor-release weight parameter for calibrated downstream impact;
       uses median-MAD normalization throughout for outlier resilience and sign preservation."""
    eps = 0.009844639816190404
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
    slack_pressure = 1.0 / (1.0 + np.exp(-4.9426312174337745 * -norm_slack))
    successor_release = 0.8836462556719553 * norm_rank * norm_work * slack_pressure
    coupled_rank = norm_rank * (1.0 + 0.5246543726832322 * slack_pressure)
    uncert_gate = np.where(norm_uncert > 0.04534090626325907, 1.0, 0.0)
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_gate
    wait_benefit = np.clip(0.6291158053089491 * ready_wait_time, 0.0, 1.0)
    duration_uncert_penalty = norm_duration * norm_uncert * slack_pressure * ddl_gate
    raw_slack_penalty = np.maximum(-slack, 0.0)
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(successor_release, -2.0, 2.0) - np.clip(coupled_rank, -2.0, 2.0) - 0.1848421867787286 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip(duration_uncert_penalty, -2.0, 2.0) + 0.9981688502784364 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 0.48306086102315327 * np.clip(norm_work, -2.0, 2.0) + np.clip(slack_pressure * 4.49550462272206, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
