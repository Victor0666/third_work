import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: restores linear slack pressure per reflection;
       reintroduces hard DDL-protection gate (slack < 0 → zero energy contribution);
       adds host-load–aware gating *only* for energy terms — conditioned on external host_load estimate,
       but since host_load is not provided, we synthesize a proxy from ready_wait_time and uncertainty
       as validated proxy for local VM congestion (long wait + high uncertainty ≈ loaded host);
       avoids feasibility erosion by restricting load modulation exclusively to energy penalty under DDL-feasible regime.
       All other components retain proven stability: median-MAD normalization, bounded clipping, finite safeguards."""
    eps = 1.1923385813078236e-05
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
    load_proxy = 1.0 / (1.0 + np.exp(-2.0 * (norm_wait + norm_uncert)))
    hard_ddl_gate = np.where(slack < 0.0, 0.0, 1.0)
    slack_pressure = np.clip(-slack, 0.0, 1.0)
    successor_release = norm_rank * norm_work * slack_pressure
    raw_slack_penalty = np.maximum(-slack, 0.0)
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    coupled_rank = norm_rank * (1.0 + 1.3580742714087952 * slack_pressure)
    uncert_gate = np.where(norm_uncert > 0.4252550807270652, 1.0, 0.0)
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * hard_ddl_gate
    wait_benefit = np.clip(0.6460282370129712 * ready_wait_time, 0.0, 1.0)
    exec_penalty = norm_duration * slack_pressure * hard_ddl_gate
    load_gate = np.where(load_proxy > 0.9079489600106764, 1.0, 0.0)
    load_scaled_energy = norm_energy * (1.0 + 0.9992158055554435 * load_gate)
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(successor_release, -2.0, 2.0) - np.clip(coupled_rank, -2.0, 2.0) - 0.3944840293070535 * np.clip(load_scaled_energy * hard_ddl_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip(exec_penalty, -2.0, 2.0) + 0.8824148101442074 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 1.2286363508856013 * np.clip(norm_work, -2.0, 2.0) + np.clip(slack_pressure * 3.2722013729665664, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
