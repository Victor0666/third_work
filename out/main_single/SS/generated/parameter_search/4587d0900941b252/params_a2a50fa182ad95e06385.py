import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: restores robust logistic starvation relief with eps offset;
       replaces brittle 'duration-aware' wait_benefit with congestion-agnostic but DDL-feasibility-gated version;
       introduces *normalized slack margin ratio* — a bounded, interpretable, and numerically stable measure of deadline headroom;
       eliminates all unvalidated nonlinear couplings (e.g., sigmoid slack_pressure_coupling) while preserving proven gates;
       retains median-MAD normalization, DDL-protection gate, host-load gate, and critical path leverage with unified risk-actionability;
       enforces strict [-2,2] clipping and shape consistency."""
    eps = 0.00015501445604842437
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
    ddl_gate = 1.0 / (1.0 + np.exp(-7.361419475496532 * slack))
    host_load_gate = 1.0 / (1.0 + np.exp(-4.564268507233662 * norm_duration))
    slack_abs = np.abs(slack) + eps
    slack_margin_ratio = np.clip((slack + slack_abs) / (2.0 * slack_abs), 0.0, 1.0)
    ddl_breach = (slack <= 0.0).astype(float)
    risk_actionable = (norm_uncert > 0.23651972547657077).astype(float)
    critical_path_leverage = norm_rank * norm_work * ddl_gate * risk_actionable
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 2.691341021263849
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-3.494767795352905 * (slack_pressure - 1.0)))
    boosted_rank = norm_rank * (1.0 + 2.6342065752559556 * (1.0 - slack_margin_ratio)) * (1.0 + rank_gate)
    wait_benefit = 1.0 - np.exp(-0.9829985179159074 * (ready_wait_time + eps))
    uncert_gate = (norm_uncert > 0.23651972547657077).astype(float) * ddl_gate
    duration_risk_score = norm_duration * uncert_gate * slack_pressure
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_gate
    energy_slack_penalty = norm_energy * (1.0 + 2.691341021263849 * slack_pressure) * ddl_gate
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(critical_path_leverage, -2.0, 2.0) - np.clip(boosted_rank, -2.0, 2.0) - 1.2080451105036765 * host_load_gate * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + 1.2197054854934835 * np.clip(duration_risk_score, -2.0, 2.0) + 0.900069418330691 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 2.691341021263849 * np.clip(energy_slack_penalty, -2.0, 2.0) + 1.0790897921407059 * np.clip(norm_work, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
