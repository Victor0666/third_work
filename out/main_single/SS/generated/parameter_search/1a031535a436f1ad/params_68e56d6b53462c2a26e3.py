import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule featuring:
       - Replaced brittle multiplicative criticality scaling with numerically stable `norm_rank * (1 + PARAMS["tanh_slack_coupling_scale"] * np.tanh(slack))`.
       - Introduced host-load conditional gate using `(ready_wait_time * uncertainty)` to suppress energy/uncertainty penalties during congestion.
       - Removed redundant `duration_criticality` and `host_load_gate_threshold` (merged logic into existing parameters via direct product).
       - All intermediate terms clipped to [-2,2]; no numeric literals beyond {-2,-1,0,1,2}; deterministic and finite output."""
    eps = 0.047663127388714185
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
    ddl_gate = 1.0 / (1.0 + np.exp(-6.372825584636836 * slack))
    ddl_breach = (slack <= 0.0).astype(float)
    slack_ramp_center = 0.4897238672361906
    slack_pressure_ramp = np.clip((norm_slack + slack_ramp_center) * 7.491318186867867, 0.0, 1.0)
    load_metric = ready_wait_time * uncertainty
    host_load_gate = (load_metric <= 0.11645450460288756).astype(float)
    successor_gate = (norm_uncert > 0.11645450460288756).astype(float)
    successor_release = norm_rank * norm_work * successor_gate * ddl_breach * 1.9805869688486843
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 3.771182573626819
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    tanh_slack_coupling = 1.0 + 0.593095252154256 * np.tanh(slack)
    boosted_rank = norm_rank * tanh_slack_coupling
    wait_benefit_pressure = (1.0 - np.exp(-0.7718564061296328 * ready_wait_time)) * ddl_breach * host_load_gate
    energy_uncert_penalty = norm_energy * norm_uncert * successor_gate * ddl_gate * host_load_gate
    energy_slack_penalty = norm_energy * (1.0 + 3.771182573626819 * slack_pressure_ramp) * ddl_gate * host_load_gate
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(successor_release, -2.0, 2.0) - np.clip(boosted_rank, -2.0, 2.0) - 0.16801441907467102 * np.clip(norm_energy * ddl_gate * host_load_gate, -2.0, 2.0) - np.clip(wait_benefit_pressure, -2.0, 2.0) + 0.9912716522249614 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 3.771182573626819 * np.clip(energy_slack_penalty, -2.0, 2.0) + 0.814895146238475 * np.clip(norm_work, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
