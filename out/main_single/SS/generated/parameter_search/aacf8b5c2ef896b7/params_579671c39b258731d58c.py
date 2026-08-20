import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule: combines Parent 2's DDL-protection gate and bounded interactions with Parent 1's validated critical-path leverage;
       uses median-MAD normalization for robustness; clips all terms to [-2,2]; preserves monotonic slack pressure;
       removes redundant 'critical_path_leverage_weight' to comply with 12-parameter limit while retaining its effect via direct scaling."""
    eps = 3.355455251969459e-06
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
    ddl_gate = 1.0 / (1.0 + np.exp(-6.178495244473324 * slack))
    ddl_breach = (slack <= 0.0).astype(float)
    critical_path_leverage = norm_rank * norm_work * ddl_breach
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 2.610573929393688
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    coupled_slack = np.clip(norm_slack, -1.0, 1.0)
    rank_slack_coupling = 1.0 + 0.8738461861778252 * coupled_slack
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-1.7405974056573896 * (slack_pressure - 1.0)))
    boosted_rank = norm_rank * rank_slack_coupling * (1.0 + 0.8738461861778252 * rank_gate)
    uncert_gate = 1.0 / (1.0 + np.exp(-6.178495244473324 * (norm_uncert - 0.25831491074993823)))
    duration_risk_score = norm_duration * uncert_gate * slack_pressure * ddl_gate
    wait_benefit = 1.0 - np.exp(-0.06581100457245581 * (ready_wait_time + 2.1664634256964266e-05))
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_gate
    energy_slack_penalty = norm_energy * (1.0 + 2.610573929393688 * slack_pressure) * ddl_gate
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(critical_path_leverage, -2.0, 2.0) - np.clip(boosted_rank, -2.0, 2.0) - 0.6317784733574485 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + 0.05528862138871485 * np.clip(duration_risk_score, -2.0, 2.0) + 0.29122711816882063 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 2.610573929393688 * np.clip(energy_slack_penalty, -2.0, 2.0) + 0.6009094254116569 * np.clip(norm_work, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
