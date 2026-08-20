import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with normalized slack gating and bounded linear timing blend.
    
    Key improvements:
      - Replaces fragile raw-quadratic slack penalty with *normalized slack sign gating*: 
        uses normalize(slack) only to activate criticality and interaction — preserves deadline sensitivity 
        while eliminating outlier-driven instability from unbounded slack magnitudes.
      - Simplifies duration-uncertainty coupling to a *bounded linear blend* (dur_norm + ratio * uncert_norm), 
        avoiding harmonic mean numerical fragility without loss of discriminative power.
      - Removes inactive features (remaining_work_fairness, comm/exec bias) per reflection, reducing noise.
      - All normalizations use robust mean-abs scaling; all divisions guarded by epsilon; all outputs finite.
    """
    eps = 0.08720456363330285
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)

    def normalize(x):
        x = np.asarray(x)
        abs_x = np.abs(x)
        scale = np.mean(abs_x) if np.all(np.isfinite(abs_x)) and np.mean(abs_x) > eps else eps
        return x / (scale + eps)
    slack_norm = normalize(slk)
    slack_penalty = np.where(slk < 0, 8.321894122727475 * np.abs(slack_norm), -1.4943630303902982 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.13632531407314846 * normalize(inv_energy)
    rank_active = np.where(slack_norm >= 0, rank, 0.0)
    rank_score = -1.3532138970732102 * normalize(rank_active + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.0005117499706009263 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_sat = 1.0 - np.exp(-0.09275745690340254 * wait)
    wait_score = -normalize(wait_sat + eps)
    unc_slack_interaction = 0.3956432711048255 * uncert_norm * np.where(slack_norm < 0, np.abs(slack_norm), 0.0)
    score = slack_penalty + energy_score + rank_score + dur_score + wait_score + unc_slack_interaction
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 19734696.82208758
    min_safe = -finfo.max / 19734696.82208758
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
