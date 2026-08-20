import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule: combines Parent 2's DDL-protection gate with Parent 1's bounded linear starvation relief,
       uses unified MAD-based robust normalization, and enforces strict gating of all energy/uncertainty terms by ddl_gate.
       Exactly 12 parameters; no numeric literals except -2,-1,0,1,2; shape (N,) guaranteed."""
    eps = 8.977442535720845e-05
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
        x = np.asarray(x)
        if N == 1:
            center = x[0]
            spread = eps
        else:
            center = np.median(x)
            spread = np.median(np.abs(x - center))
        return (x - center) / (spread + eps)
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.2308904856901581
    norm_slack_penalty = robust_normalize(raw_slack_penalty)
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-5.353732870684363 * (slack_pressure - 1.0)))
    boosted_rank = norm_rank * (1.0 + 1.843420845358005 * rank_gate)
    ddl_gate = 1.0 / (1.0 + np.exp(-6.682381933858923 * slack))
    uncert_gate = 1.0 / (1.0 + np.exp(-6.682381933858923 * (norm_uncert - 0.30752309489901153)))
    duration_risk_score = norm_duration * uncert_gate * slack_pressure * ddl_gate
    wait_benefit_raw = 0.5988713771844777 * (ready_wait_time + 2.986834560057692e-08)
    wait_benefit = np.clip(wait_benefit_raw, 0.0, 1.0)
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_gate
    energy_slack_penalty = norm_energy * (1.0 + 1.2308904856901581 * slack_pressure) * ddl_gate
    score = +norm_slack_penalty - boosted_rank - 0.660783884279527 * norm_energy * ddl_gate - wait_benefit + 0.7621445011956602 * duration_risk_score + 0.0623014713288465 * energy_uncert_penalty + 1.2308904856901581 * energy_slack_penalty + 0.7551994156241174 * norm_work
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
