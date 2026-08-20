import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule: merges Parent 2's stability and gating with Parent 1's successor-release insight;
       uses power-law slack-rank coupling and saturating anti-starvation;
       replaces per-term clipping with global score bounding;
       uses median-MAD normalization with sign-preserving centering."""
    eps = 0.005223552206413706
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
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 2.6513720255060838
    norm_slack_penalty = robust_normalize(raw_slack_penalty)
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_coupling_factor = (1.0 + slack_pressure) ** 2.6513720255060838
    boosted_rank = norm_rank * rank_coupling_factor
    ddl_gate = 1.0 / (1.0 + np.exp(-4.840119574784705 * slack))
    uncert_gate = np.where(norm_uncert > 0.12702160063226325, 1.0, 0.0)
    duration_risk_score = norm_duration * norm_uncert * uncert_gate * ddl_gate
    wait_benefit = 1.0 - np.exp(-0.6079119454797227 * (norm_wait + 6.880655654579515e-06) ** 1.0)
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_gate
    energy_slack_penalty = norm_energy * (1.0 + slack_pressure * 2.6513720255060838) * ddl_gate
    successor_release_score = norm_rank * norm_work * (slack <= 0.0).astype(float)
    score = +norm_slack_penalty - successor_release_score - boosted_rank - 1.004695105594917 * norm_energy * ddl_gate - wait_benefit + 0.010153676399700277 * duration_risk_score + 0.4975091807670634 * energy_uncert_penalty + 2.6513720255060838 * energy_slack_penalty + 0.6543045008609971 * norm_work
    score = np.clip(score, -4.960430941475801, 10.248550217314431)
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.reshape(-1)
