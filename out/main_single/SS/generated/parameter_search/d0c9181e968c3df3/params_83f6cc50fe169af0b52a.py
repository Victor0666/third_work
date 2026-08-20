import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule: merges Parent 2's stability and gating with Parent 1's successor-release insight;
       uses power-law slack-rank coupling and saturating anti-starvation;
       replaces per-term clipping with global score bounding;
       uses median-MAD normalization with sign-preserving centering."""
    eps = 7.58311291606759e-05
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
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 3.768122557862451
    norm_slack_penalty = robust_normalize(raw_slack_penalty)
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_coupling_factor = (1.0 + slack_pressure) ** 3.768122557862451
    boosted_rank = norm_rank * rank_coupling_factor
    ddl_gate = 1.0 / (1.0 + np.exp(-3.3839207352557565 * slack))
    uncert_gate = np.where(norm_uncert > 0.6625950605261175, 1.0, 0.0)
    duration_risk_score = norm_duration * norm_uncert * uncert_gate * ddl_gate
    wait_benefit = 1.0 - np.exp(-0.21972813223629603 * (norm_wait + 2.3039558697459436e-06) ** 1.0)
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_gate
    energy_slack_penalty = norm_energy * (1.0 + slack_pressure * 3.768122557862451) * ddl_gate
    successor_release_score = norm_rank * norm_work * (slack <= 0.0).astype(float)
    score = +norm_slack_penalty - successor_release_score - boosted_rank - 0.6039640827380937 * norm_energy * ddl_gate - wait_benefit + 0.7342320038165903 * duration_risk_score + 0.2987891439129078 * energy_uncert_penalty + 3.768122557862451 * energy_slack_penalty + 0.0012848350222435271 * norm_work
    score = np.clip(score, -6.239947101438096, 15.337096845046467)
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.reshape(-1)
