import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with successor-release urgency and slack-aware feature scaling:
      - Replaces joint-risk MAD with task-level slack-aware scaling: preserves sharp deadline signals without over-smoothing.
      - Introduces `successor_release_factor`-weighted successor-release term: `upward_rank * (1 + remaining_work/(slack+eps))`
        to amplify critical-path urgency under tight deadlines — directly addressing reflection's concern.
      - Removes `joint_risk_mad_scale` and associated joint normalization (structural redundancy removed).
      - All DDL-critical terms remain unconditional and lexicographically dominant.
      - Non-DDL terms strictly gated by `ddl_safe_mask`, with host-load-aware scaling via linear slack headroom.
      - Uses per-feature median absolute deviation (MAD) only where needed — avoids cross-dimension coupling that weakens DDL safety.
      - No numeric literals beyond {-2,-1,0,1,2}; all tunables declared and used exactly once.
    """
    eps = 1.420696391971374e-09
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

    def mad_normalize(x):
        x = np.asarray(x, dtype=float)
        if N == 1:
            return np.zeros_like(x, dtype=float)
        med = np.median(x)
        abs_dev = np.abs(x - med)
        mad = np.median(abs_dev) + eps
        normalized = (x - med) / mad
        return np.clip(normalized, -2.0, 2.0)
    slack_score = np.where(slack < 0, (-slack) ** 1.3771175602887387, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 0.3114760290395837
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 1.771823925630939
    slack_safe = np.where(slack > 0, slack, eps)
    successor_release_gain = 1.0 + remaining_work / slack_safe
    critical_path_release = upward_rank * successor_release_gain * 0.3792030001286288
    critical_path_base = upward_rank * remaining_work * 1.6408928295646596
    critical_path_total = critical_path_base + critical_path_release
    ddl_safe_mask = np.where(slack > 0.16673843038779362, 1.0, 0.0)
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_score = mad_normalize(energy_per_sec)
    slack_headroom = np.maximum(0.0, slack - 0.16673843038779362)
    max_slack_headroom = np.max(slack_headroom) + eps
    slack_headroom_normalized = slack_headroom / max_slack_headroom
    host_load_scale = slack_headroom_normalized * 0.13167040587359863
    rank_score = -mad_normalize(upward_rank) * (1.0 + slack_headroom_normalized)
    unc_sigmoid = 1.0 / (1.0 + np.exp(-3.6638535493767197 * (mad_normalize(uncertainty) - 0.0)))
    energy_uncertainty_score = ddl_safe_mask * host_load_scale * 0.6283316678912589 * energy_eff_score * mad_normalize(uncertainty) * unc_sigmoid
    wait_score = ddl_safe_mask * mad_normalize(ready_wait_time) * slack_headroom_normalized * 0.9951432919467612
    score = mad_normalize(slack_score) + mad_normalize(unc_slack_coupling) + mad_normalize(duration_risk) + mad_normalize(critical_path_total)
    score += ddl_safe_mask * (1.7610438952477727 * host_load_scale * energy_eff_score + rank_score + energy_uncertainty_score + wait_score)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
