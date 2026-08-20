import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule featuring:
       - Critical-path-aware successor-release interaction: replaces adaptive uncertainty gating with direct
         upward_rank × remaining_work coupling, activated only under deadline pressure and gated by uncertainty.
       - Bounded piecewise-linear slack-pressure gate (ramp) instead of sigmoid: improves interpretability and
         avoids overfitting near zero slack; uses clip-based ramp with tunable offset and steepness.
       - Removed duration_robustness and wait_saturation_offset per diagnostics: eliminates noise and over-parameterization.
       - All normalization remains robust median-MAD; all terms clipped to [-2,2]; final score finite and shape-(N,)."""
    eps = 0.04924475572504591
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
    ddl_gate = 1.0 / (1.0 + np.exp(-2.2640563920679155 * slack))
    ddl_breach = (slack <= 0.0).astype(float)
    slack_ramp_center = 0.31603735653106724
    slack_pressure_ramp = np.clip((norm_slack + slack_ramp_center) * 4.666426365997568, 0.0, 1.0)
    successor_gate = (norm_uncert > 0.20915844890209767).astype(float)
    successor_release = norm_rank * norm_work * successor_gate * ddl_breach * 0.27564730130566806
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 2.1972989984797286
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    coupled_slack = np.clip(norm_slack, -1.0, 1.0)
    rank_slack_coupling = 1.0 + 2.779581782752434 * coupled_slack
    boosted_rank = norm_rank * rank_slack_coupling * (1.0 + slack_pressure_ramp * 2.779581782752434)
    wait_benefit_pressure = (1.0 - np.exp(-0.7676567290971033 * ready_wait_time)) * ddl_breach
    energy_uncert_penalty = norm_energy * norm_uncert * successor_gate * ddl_gate
    energy_slack_penalty = norm_energy * (1.0 + 2.1972989984797286 * slack_pressure_ramp) * ddl_gate
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(successor_release, -2.0, 2.0) - np.clip(boosted_rank, -2.0, 2.0) - 1.9480852472949892 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - np.clip(wait_benefit_pressure, -2.0, 2.0) + 0.8743424763339125 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 2.1972989984797286 * np.clip(energy_slack_penalty, -2.0, 2.0) + 1.1675088504480864 * np.clip(norm_work, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
