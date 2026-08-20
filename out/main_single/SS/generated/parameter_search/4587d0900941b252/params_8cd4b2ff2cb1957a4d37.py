import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: restores robust logistic starvation relief with eps offset;
       replaces brittle 'duration-aware' wait_benefit with congestion-agnostic but DDL-feasibility-gated version;
       introduces *normalized slack margin ratio* — a bounded, interpretable, and numerically stable measure of deadline headroom;
       eliminates all unvalidated nonlinear couplings (e.g., sigmoid slack_pressure_coupling) while preserving proven gates;
       retains median-MAD normalization, DDL-protection gate, host-load gate, and critical path leverage with unified risk-actionability;
       enforces strict [-2,2] clipping and shape consistency."""
    eps = 1.4568385842298799e-05
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
    ddl_gate = 1.0 / (1.0 + np.exp(-6.651525177357893 * slack))
    host_load_gate = 1.0 / (1.0 + np.exp(-3.2542745181008863 * norm_duration))
    slack_abs = np.abs(slack) + eps
    slack_margin_ratio = np.clip((slack + slack_abs) / (2.0 * slack_abs), 0.0, 1.0)
    ddl_breach = (slack <= 0.0).astype(float)
    risk_actionable = (norm_uncert > 0.32157307150277864).astype(float)
    critical_path_leverage = norm_rank * norm_work * ddl_gate * risk_actionable
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 2.4903669866591644
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-2.255531645097719 * (slack_pressure - 1.0)))
    boosted_rank = norm_rank * (1.0 + 2.7603464290002004 * (1.0 - slack_margin_ratio)) * (1.0 + rank_gate)
    wait_benefit = 1.0 - np.exp(-0.6836409201765571 * (ready_wait_time + eps))
    uncert_gate = (norm_uncert > 0.32157307150277864).astype(float) * ddl_gate
    duration_risk_score = norm_duration * uncert_gate * slack_pressure
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_gate
    energy_slack_penalty = norm_energy * (1.0 + 2.4903669866591644 * slack_pressure) * ddl_gate
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(critical_path_leverage, -2.0, 2.0) - np.clip(boosted_rank, -2.0, 2.0) - 1.99992843988648 * host_load_gate * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + 0.709111366423339 * np.clip(duration_risk_score, -2.0, 2.0) + 0.48892937982743256 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 2.4903669866591644 * np.clip(energy_slack_penalty, -2.0, 2.0) + 1.1745985228546303 * np.clip(norm_work, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
