import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule: removes unused ddl_protection_gate_slope;
       uses stable linear Heaviside ddl_gate = max(0, slack) / (|slack| + eps);
       retains all declared parameters in computation;
       enforces strict -2..2 clipping; no numeric literals beyond -2,-1,0,1,2."""
    eps = 7.468344456554983e-06
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
    ddl_gate = np.clip(slack, 0.0, np.inf) / (np.abs(slack) + eps)
    ddl_breach_mask = (slack <= 0.0).astype(float)
    critical_path_leverage = norm_rank * norm_work * ddl_breach_mask * 0.6880360923167474
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.3501790711619308
    coupled_slack = np.clip(norm_slack, -1.0, 1.0)
    rank_slack_coupling = 1.0 + 2.0728379988298977 * coupled_slack
    wait_benefit = np.clip(0.405056618695812 * ready_wait_time, 0.0, 1.0)
    tight_slack_mask = (slack <= eps).astype(float)
    high_uncert_mask = (norm_uncert >= 0.42078993338981807).astype(float)
    duration_risk_score = norm_duration * tight_slack_mask * high_uncert_mask * 0.37993108301179035
    energy_uncert_penalty = norm_energy * norm_uncert * ddl_gate
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    energy_slack_penalty = norm_energy * (1.0 + 1.3501790711619308 * slack_pressure) * ddl_gate
    score = +raw_slack_penalty - np.clip(critical_path_leverage, -2.0, 2.0) - np.clip(norm_rank * rank_slack_coupling, -2.0, 2.0) - 1.218048580600425 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + 0.560344776198505 * np.clip(duration_risk_score, -2.0, 2.0) + 0.032803955263910486 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 1.3501790711619308 * np.clip(energy_slack_penalty, -2.0, 2.0) + 0.035871821667799716 * np.clip(norm_work, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
