import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule: combines Parent 2's robust DDL protection and load gating with Parent 1's successor-release interaction;
       replaces hard clipping with bounded sigmoid coupling for slack-pressure modulation;
       introduces duration-aware starvation relief decoupled from congestion;
       adds risk-gated critical-path leverage requiring both DDL feasibility AND actionable uncertainty;
       retains median-MAD normalization, all validated gates, and strict numeric literal constraints."""
    eps = 0.08998486746984938
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
    ddl_gate = 1.0 / (1.0 + np.exp(-7.834852439118018 * slack))
    host_load_gate = 1.0 / (1.0 + np.exp(-6.104767422130655 * norm_duration))
    slack_pressure_coupling = 2.0 / (1.0 + np.exp(-2.0 * norm_slack)) - 1.0
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-4.793518556877952 * (slack_pressure - 1.0)))
    ddl_breach = (slack <= 0.0).astype(float)
    risk_actionable = (norm_uncert > 0.7013685156733283).astype(float)
    critical_path_leverage = norm_rank * norm_work * ddl_gate * risk_actionable
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.4448683998019916
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    successor_release = min_exec_time * norm_rank * ddl_breach
    wait_benefit = (1.0 - np.exp(-0.04424319478589052 * ready_wait_time)) * (1.0 - host_load_gate)
    uncert_gate = (norm_uncert > 0.7013685156733283).astype(float) * ddl_gate
    duration_risk_score = norm_duration * uncert_gate * slack_pressure
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_gate
    energy_slack_penalty = norm_energy * (1.0 + 1.4448683998019916 * slack_pressure) * ddl_gate
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(critical_path_leverage, -2.0, 2.0) - np.clip(successor_release, -2.0, 2.0) - 1.1726697070320673 * host_load_gate * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + 0.8903922090336406 * np.clip(duration_risk_score, -2.0, 2.0) + 0.15266146191474725 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 1.4448683998019916 * np.clip(energy_slack_penalty, -2.0, 2.0) + 0.8174164847159563 * np.clip(norm_work, -2.0, 2.0) + 2.7979014654170884 * np.clip(norm_rank * slack_pressure_coupling, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
