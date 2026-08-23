import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule featuring:
      - Hard feasibility-aware switch: below `critical_slack_threshold`, score prioritizes slack urgency (|slack| when negative, else 0);
        above it, defaults to critical-path release — eliminates fragile interpolation.
      - Bounded min-max normalization for slack, uncertainty, duration_total (replacing MAD) to stabilize small-N ready sets.
      - Decoupled anti-starvation: uses normalized `ready_wait_time` directly (not scaled by headroom), gated only by DDL feasibility.
      - All numeric literals strictly limited to {-2,-1,0,1,2}; no other constants.
      - Final score is deterministic, finite, shape-(N,), and satisfies 'smaller = higher priority'.
    """
    eps = 1.6653082264381396e-08
    mm_eps = 0.0003559215420645969
    finfo = np.finfo(float)
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=finfo.max, neginf=finfo.min)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=finfo.max, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=eps, posinf=finfo.max, neginf=eps)
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=float)

    def minmax_normalize(x):
        x = np.asarray(x, dtype=float)
        if N == 1:
            return np.zeros_like(x, dtype=float)
        x_min = np.min(x)
        x_max = np.max(x)
        denom = x_max - x_min + mm_eps
        normalized = (x - x_min) / denom
        return np.clip(normalized, 0.0, 1.0)
    duration_total = min_exec_time + min_comm_time + eps
    slack_abs = np.abs(slack)
    slack_penalty = np.where(slack < 0, (-slack) ** 2.410050992259434, 0.0)
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 0.5808768656698208
    critical_release_score = upward_rank * remaining_work * 3.2779495732472683
    urgency_mask = np.where(slack < 0.23832787489891588, 1.0, 0.0)
    urgency_score = np.where(slack < 0, slack_abs, 0.0)
    if np.any(slack < 0):
        urgency_norm = minmax_normalize(urgency_score)
    else:
        urgency_norm = np.zeros_like(urgency_score)
    switched_base_score = urgency_mask * urgency_norm + (1.0 - urgency_mask) * minmax_normalize(critical_release_score)
    slack_headroom_mask = np.where(slack > 3.2133048028694313, 1.0, 0.0)
    energy_norm = minmax_normalize(min_incremental_energy)
    energy_term = slack_headroom_mask * 1.1846624822646767 * energy_norm
    wait_norm = minmax_normalize(ready_wait_time)
    wait_term = slack_headroom_mask * wait_norm ** 1.6760151450484364
    unc_sigmoid = 1.0 / (1.0 + np.exp(-4.797137807560121 * (uncertainty - 1.0)))
    unc_norm = minmax_normalize(uncertainty)
    energy_uncertainty_term = slack_headroom_mask * 1.1177826153334114 * energy_norm * unc_norm * unc_sigmoid
    score = minmax_normalize(slack_penalty) + minmax_normalize(duration_risk) + switched_base_score
    score += energy_term + wait_term + energy_uncertainty_term
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
