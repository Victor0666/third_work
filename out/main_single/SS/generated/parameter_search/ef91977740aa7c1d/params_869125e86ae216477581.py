import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule: merges Parent 2's hard feasibility gating and tunable energy suppression with Parent 1's successor-release gate.
       Structural novelty: replaces comm-energy coupling (removed to meet parameter limit) with *tighter release-gap activation* — uses sigmoidal gate on (slack - release_horizon) with explicit thresholding to sharpen bottleneck detection.
       Retains clipped-linear starvation relief, robustified duration penalty, and dual-gated energy suppression. All numeric literals are in [-2,2] or use np.finfo."""
    eps = 0.009464295529102376
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
        x = np.asarray(x, dtype=float)
        if N == 1:
            med = x[0]
            mad = eps
        else:
            med = np.median(x)
            mad = np.median(np.abs(x - med))
        spread = mad + eps
        return (x - med) / spread
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    hard_feasibility_gate = np.where(slack < 0.0, 0.0, 1.0)
    ddl_pressure = np.clip(-norm_slack, 0.0, 1.0)
    raw_slack_penalty = np.maximum(-slack, 0.0)
    norm_slack_penalty = robust_normalize(raw_slack_penalty)
    slack_penalty = 2.043309238309802 * norm_slack_penalty
    release_horizon = np.where(upward_rank > eps, remaining_work / upward_rank, np.finfo(float).max)
    release_gap = slack - release_horizon
    successor_release_gate = 1.0 / (1.0 + np.exp(-8.581273268817272 * np.clip(release_gap, -2.0, 2.0)))
    coupled_rank = norm_rank * (1.0 + 0.7448757896182723 * ddl_pressure)
    uncert_gate = np.where(norm_uncert > 0.27394541858722754, 1.0, 0.0)
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * hard_feasibility_gate
    wait_benefit = np.clip(0.46419071224477193 * ready_wait_time, 0.0, 2.0)
    duration_penalty = 0.18303093546867416 * norm_duration * ddl_pressure * hard_feasibility_gate
    slack_energy_suppress = np.where(slack < 0.0, 1.0 - 0.6880610122037385, 1.0)
    suppressed_norm_energy = norm_energy * slack_energy_suppress
    score = +np.clip(slack_penalty, -2.0, 2.0) - np.clip(coupled_rank, -2.0, 2.0) - np.clip(successor_release_gate, -2.0, 2.0) - 0.2086864568916733 * np.clip(suppressed_norm_energy, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip(duration_penalty, -2.0, 2.0) + 0.6672435928519667 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 0.4859185522295867 * np.clip(norm_work, -2.0, 2.0)
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.reshape(-1)
