import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule:
       - Restores deadline-pressure-only activation for criticality coupling (removes over-constraining congestion gate).
       - Replaces fixed uncertainty_gate_threshold with slack-aware variant: active only when DDL-feasible AND uncertainty > base * (1 - sigmoid(slack)).
       - Introduces dedicated wait_benefit_under_pressure term: linear starvation relief *exclusively* for slack <= 0 tasks.
       - Simplifies congestion modeling: uses raw uncertainty * (slack >= 0) as direct risk signal instead of composite congestion.
       - All terms bounded via [-2,2] clipping; final score deterministic, finite, shape-(N,)."""
    eps = 0.09638269785880758
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def quantile_normalize(x):
        x = np.copy(x)
        if N == 1:
            q1 = q3 = x[0]
        else:
            q1 = np.quantile(x, 0.23389631121300264)
            q3 = np.quantile(x, 0.7379707234963105)
        iqr = q3 - q1 if q3 > q1 else eps
        return (x - q1) / iqr
    norm_slack = quantile_normalize(slack)
    norm_energy = quantile_normalize(min_incremental_energy)
    norm_duration = quantile_normalize(min_exec_time + min_comm_time)
    norm_rank = quantile_normalize(upward_rank)
    norm_work = quantile_normalize(remaining_work)
    norm_wait = quantile_normalize(ready_wait_time)
    norm_uncert = quantile_normalize(uncertainty)
    slack_med = np.median(slack)
    slack_mad = np.median(np.abs(slack - slack_med))
    slack_spread = slack_mad if slack_mad > eps else eps
    norm_slack_mad = (slack - slack_med) / slack_spread
    ddl_feasible = (slack >= 0.0).astype(float)
    ddl_feasible_gate = 1.0 / (1.0 + np.exp(-2.6023749208042863 * (slack - 0.0)))
    ddl_pressure = (slack <= 0.0).astype(float)
    adaptive_uncert_threshold = 0.16114891846069146 * (1.0 - 1.0 / (1.0 + np.exp(-slack)))
    slack_aware_uncert_gate = (uncertainty > adaptive_uncert_threshold).astype(float) * ddl_feasible
    critical_coupling = norm_rank * norm_work * ddl_pressure * 0.9218632327240719
    wait_benefit_pressure = np.clip(0.6650031248112848 * ready_wait_time, 0.0, 2.0) * ddl_pressure
    energy_uncert_coupling = norm_energy * norm_uncert * slack_aware_uncert_gate
    slack_pressure = np.clip(-norm_slack_mad, 0.0, 2.0)
    energy_slack_amplifier = 1.0 + 1.1328783791816122 * slack_pressure
    risk_signal = norm_uncert * ddl_feasible
    score = +np.clip(norm_slack_mad, -2.0, 2.0) - np.clip(critical_coupling, -2.0, 2.0) - np.clip(norm_rank, -2.0, 2.0) - np.clip(wait_benefit_pressure, -2.0, 2.0) - 0.4144386241520224 * np.clip(norm_energy * ddl_feasible_gate, -2.0, 2.0) + 0.9892070343881803 * np.clip(energy_uncert_coupling, -2.0, 2.0) + np.clip(norm_energy * energy_slack_amplifier * ddl_feasible, -2.0, 2.0) + 0.16338435188182615 * np.clip(norm_work * ddl_pressure, -2.0, 2.0) + np.clip(risk_signal, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
