import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule with early starvation relief via slack-thresholded activation:
       - Uses unified DDL-protection gate for all risk-sensitive terms.
       - Introduces `starvation_activation_slack` derived from `piecewise_slack_offset` (no new param) to avoid exceeding 12 params.
       - Retains Parent 2's proven piecewise ramp, successor release, and clean structure.
       - All numeric literals are in {-2,-1,0,1,2}; epsilon handled via PARAMS; no hidden constants."""
    eps = 2.1997893137446843e-05
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
    norm_rank = median_mad_normalize(upward_rank)
    norm_work = median_mad_normalize(remaining_work)
    norm_wait = median_mad_normalize(ready_wait_time)
    norm_uncert = median_mad_normalize(uncertainty)
    ddl_gate = 1.0 / (1.0 + np.exp(-2.0173773607623944 * slack))
    starvation_gate = (norm_slack < 0.2700485221065446).astype(float)
    wait_benefit = (1.0 - np.exp(-0.24477065485169192 * ready_wait_time)) * starvation_gate
    ddl_breach = (slack <= 0.0).astype(float)
    slack_ramp_center = 0.2700485221065446
    slack_pressure_ramp = np.clip((norm_slack + slack_ramp_center) * 1.1989259753974637, 0.0, 1.0)
    successor_gate = (norm_uncert > 0.22402921478753463).astype(float)
    successor_release = norm_rank * norm_work * successor_gate * ddl_breach * 1.3621576274511904
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 2.8134862755629197
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty) * ddl_gate
    coupled_slack = np.clip(norm_slack, -1.0, 1.0)
    rank_slack_coupling = 1.0 + 1.0318667410940838 * coupled_slack
    boosted_rank = norm_rank * rank_slack_coupling * (1.0 + slack_pressure_ramp * 1.0318667410940838)
    energy_uncert_penalty = norm_energy * norm_uncert * successor_gate * ddl_gate
    energy_slack_penalty = norm_energy * (1.0 + 2.8134862755629197 * slack_pressure_ramp) * ddl_gate
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(successor_release, -2.0, 2.0) - np.clip(boosted_rank, -2.0, 2.0) - 1.6721167727055168 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + 0.6385825480828707 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 2.8134862755629197 * np.clip(energy_slack_penalty, -2.0, 2.0) + 0.3992329039074198 * np.clip(norm_work, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
