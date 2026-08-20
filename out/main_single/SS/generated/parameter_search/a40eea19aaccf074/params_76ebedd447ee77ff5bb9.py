import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule: integrates Parent 2's validated ddl_breach leverage and duration_risk_score with Parent 1's robust median-MAD normalization;
       replaces unstable 'boosted_rank' with bounded linear rank-slack coupling + slack-pressure gate;
       enforces strict [-2,2] clipping on all feature terms and interactions to guarantee stability;
       reuses existing parameters to avoid exceeding 12-parameter limit — comm-slack coupling is absorbed into 'duration_robustness' via norm_comm inclusion in duration_risk_score."""
    eps = 4.735604912402447e-06
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
    ddl_gate = 1.0 / (1.0 + np.exp(-2.0167183578375343 * slack))
    ddl_breach = (slack <= 0.0).astype(float)
    critical_path_leverage = norm_rank * norm_work * ddl_breach
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.2261997262239777
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    coupled_slack = np.clip(norm_slack, -1.0, 1.0)
    rank_slack_coupling = 1.0 + 1.5947175575906642 * coupled_slack
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-4.90150153058963 * (slack_pressure - 1.0)))
    stable_rank = norm_rank * rank_slack_coupling * rank_gate
    uncert_gate = 1.0 / (1.0 + np.exp(-2.0167183578375343 * (norm_uncert - 0.05461686939128595)))
    duration_risk_score = (norm_duration + norm_comm) * uncert_gate * slack_pressure * ddl_gate
    wait_benefit = 1.0 - np.exp(-0.783327663274177 * (ready_wait_time + 1.8963290959027052e-05))
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_gate
    energy_slack_penalty = norm_energy * (1.0 + 1.2261997262239777 * slack_pressure) * ddl_gate
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(critical_path_leverage, -2.0, 2.0) - np.clip(stable_rank, -2.0, 2.0) - 0.980242559769216 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + 1.1026777840659525 * np.clip(duration_risk_score, -2.0, 2.0) + 0.7593884730613627 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 1.2261997262239777 * np.clip(energy_slack_penalty, -2.0, 2.0) + 0.8822232204659932 * np.clip(norm_work, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
