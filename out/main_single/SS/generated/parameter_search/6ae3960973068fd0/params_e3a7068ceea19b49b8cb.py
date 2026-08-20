import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: replaces unstable tanh-based rank-slack coupling with bounded linear interpolation;
       introduces explicit decoupled duration weight to restore adaptability; clips all intermediate terms to [-2,2]
       to prevent score explosion while preserving ordinal ranking integrity."""
    eps = 0.005740446099992667
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def robust_normalize(x):
        x_abs = np.abs(x)
        if N == 1:
            center = x_abs[0]
            spread = eps
        else:
            center = np.median(x_abs)
            spread = np.median(np.abs(x_abs - center))
        return (x_abs - center) / (spread + eps)
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.934490154864438
    norm_slack_penalty = robust_normalize(raw_slack_penalty)
    coupled_slack = np.clip(norm_slack, -1.0, 1.0)
    rank_slack_coupling = 1.0 + 1.6792575592074388 * coupled_slack
    boosted_rank = norm_rank * rank_slack_coupling
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-1.577807595620579 * (slack_pressure - 1.0)))
    boosted_rank = boosted_rank * (1.0 + 1.6792575592074388 * rank_gate)
    ddl_gate = 1.0 / (1.0 + np.exp(-2.8847968879931436 * slack))
    uncert_gate = 1.0 / (1.0 + np.exp(-2.8847968879931436 * (norm_uncert - 0.4929269559262582)))
    duration_risk_score = norm_duration * uncert_gate * slack_pressure * ddl_gate
    wait_benefit = 1.0 - np.exp(-0.19202979968539186 * (norm_wait + 4.103866657094947e-08))
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_gate
    energy_slack_penalty = norm_energy * (1.0 + 1.934490154864438 * slack_pressure) * ddl_gate
    duration_preference = 0.8585190109908848 * norm_duration * ddl_gate
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(boosted_rank, -2.0, 2.0) - 1.2908224936960917 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - duration_preference - np.clip(wait_benefit, -2.0, 2.0) + 0.8585190109908848 * np.clip(duration_risk_score, -2.0, 2.0) + 0.23937877504152968 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 1.934490154864438 * np.clip(energy_slack_penalty, -2.0, 2.0) + 0.41691446853734737 * np.clip(norm_work, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
