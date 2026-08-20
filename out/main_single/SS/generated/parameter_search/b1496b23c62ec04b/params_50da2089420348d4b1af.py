import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: replaces unstable `comm_slack_urgency` with DDL-conditioned, successor-aware critical-path booster;
       uses *tight-coupled* `upward_rank × remaining_work × ddl_gate` (not `ddl_breach`) to promote unblocking under feasibility;
       removes `duration_risk_score` and `duration_robustness` entirely to eliminate unbounded interaction and unused parameter;
       enforces strict monotonicity via clipped linear coupling and bounded sigmoid gates; all terms clipped to [-2,2]."""
    eps = 0.00022539424931770463
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
    norm_comm = median_mad_normalize(min_comm_time)
    ddl_gate = 1.0 / (1.0 + np.exp(-6.1579953324194365 * slack))
    critical_path_booster = norm_rank * norm_work * ddl_gate
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 2.9306294813889964
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    slack_pressure_gate = 1.0 / (1.0 + np.exp(-3.0739996482342815 * (slack_pressure - 1.0)))
    rank_amplifier = 1.0 + 0.8320955193395596 * slack_pressure_gate
    rank_amplifier = np.clip(rank_amplifier, 1.0, 2.0)
    boosted_rank = norm_rank * rank_amplifier
    uncert_gate = 1.0 / (1.0 + np.exp(-6.1579953324194365 * (norm_uncert - 0.053781728026964806)))
    wait_benefit = 1.0 - np.exp(-0.5072729036121272 * (ready_wait_time + 3.7759703524320454e-05))
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_gate
    energy_slack_penalty = norm_energy * (1.0 + 2.9306294813889964 * slack_pressure) * ddl_gate
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(critical_path_booster, -2.0, 2.0) - np.clip(boosted_rank, -2.0, 2.0) - 0.9522613615861854 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + 0.20007709596967166 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 2.9306294813889964 * np.clip(energy_slack_penalty, -2.0, 2.0) + 0.027870261731038362 * np.clip(norm_work, -2.0, 2.0) + np.clip(norm_comm * ddl_gate, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
