import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining robust deadline enforcement (Parent 2) with path-aware fairness (Parent 1).
    
    Key improvements:
      - Adds `remaining_work_fairness`: normalizes and penalizes tasks dominating downstream computation load,
        preventing bottlenecks in high-work subtrees.
      - Introduces `min_exec_comm_ratio_bias`: computes comm/(exec+eps) ratio to guide placement-aware urgency —
        high-ratio tasks are more sensitive to VM bandwidth selection; thus prioritized under tight slack.
      - Retains Parent 2's exponential wait decay, hard-criticality gating, and inverse-energy efficiency.
      - Replaces fragile soft coupling (Parent 1) with strict slack sign gating + interaction-only amplification.
      - Uses harmonic duration-uncertainty blend (Parent 2) for numerically stable timing signal.
      - All operations guarded against NaN/inf/zero using epsilon and np.nan_to_num with finfo-based clamping.
    """
    eps = 0.031013098220054776
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    work = np.asarray(remaining_work, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)

    def normalize(x):
        x = np.asarray(x)
        abs_x = np.abs(x)
        scale = np.mean(abs_x) if np.all(np.isfinite(abs_x)) and np.mean(abs_x) > eps else eps
        return x / (scale + eps)
    slack_penalty = np.where(slk < 0, 3.6719054760334715 * slk ** 2, -4.128622322799131 * slk)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -1.626648215573143 * normalize(inv_energy)
    rank_active = np.where(slk >= -eps, rank, 0.0)
    rank_score = -1.630944372760851 * normalize(rank_active + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = (1.0 + 0.3350127291388064 * uncert_norm) / (dur_norm + eps + 0.3350127291388064 * uncert_norm + eps)
    dur_score = normalize(dur_uncert_blend)
    wait_sat = 1.0 - np.exp(-0.1513900064419393 * wait)
    wait_score = -normalize(wait_sat + eps)
    slack_stress = np.where(slk < 0, np.abs(slk), 0.0)
    unc_slack_interaction = 3.524179895705361 * normalize(uncert) * normalize(slack_stress + eps)
    work_score = 0.541399537050878 * normalize(work)
    comm_exec_ratio = comm_t / (exec_t + eps)
    ratio_bias = np.where(slk < 0, comm_exec_ratio, 0.0)
    ratio_score = 1.0651286880455788 * normalize(ratio_bias + eps)
    score = slack_penalty + energy_score + rank_score + dur_score + wait_score + unc_slack_interaction + work_score + ratio_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 4514519.396264811
    min_safe = -finfo.max / 4514519.396264811
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
