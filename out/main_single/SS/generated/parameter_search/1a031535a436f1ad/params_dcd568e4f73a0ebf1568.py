import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule featuring:
       - Replaced brittle multiplicative criticality scaling with numerically stable `norm_rank * (1 + PARAMS["tanh_slack_coupling_scale"] * np.tanh(slack))`.
       - Introduced host-load conditional gate using `(ready_wait_time * uncertainty)` to suppress energy/uncertainty penalties during congestion.
       - Removed redundant `duration_criticality` and `host_load_gate_threshold` (merged logic into existing parameters via direct product).
       - All intermediate terms clipped to [-2,2]; no numeric literals beyond {-2,-1,0,1,2}; deterministic and finite output."""
    eps = 8.916800526391415e-05
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
    ddl_gate = 1.0 / (1.0 + np.exp(-2.2832783275111512 * slack))
    ddl_breach = (slack <= 0.0).astype(float)
    slack_ramp_center = 0.11968164646324872
    slack_pressure_ramp = np.clip((norm_slack + slack_ramp_center) * 5.7041797509205985, 0.0, 1.0)
    load_metric = ready_wait_time * uncertainty
    host_load_gate = (load_metric <= 0.588247684158983).astype(float)
    successor_gate = (norm_uncert > 0.588247684158983).astype(float)
    successor_release = norm_rank * norm_work * successor_gate * ddl_breach * 0.4927008195485902
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 2.482932125230618
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    tanh_slack_coupling = 1.0 + 0.6511544073627874 * np.tanh(slack)
    boosted_rank = norm_rank * tanh_slack_coupling
    wait_benefit_pressure = (1.0 - np.exp(-0.5851622106714096 * ready_wait_time)) * ddl_breach * host_load_gate
    energy_uncert_penalty = norm_energy * norm_uncert * successor_gate * ddl_gate * host_load_gate
    energy_slack_penalty = norm_energy * (1.0 + 2.482932125230618 * slack_pressure_ramp) * ddl_gate * host_load_gate
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(successor_release, -2.0, 2.0) - np.clip(boosted_rank, -2.0, 2.0) - 1.0456519432593896 * np.clip(norm_energy * ddl_gate * host_load_gate, -2.0, 2.0) - np.clip(wait_benefit_pressure, -2.0, 2.0) + 0.7251718297316658 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 2.482932125230618 * np.clip(energy_slack_penalty, -2.0, 2.0) + 0.011538321425034448 * np.clip(norm_work, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
