import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining robust slack gating (Parent 2) with bottleneck-aware criticality (Parent 1).
    
    Key structural improvements:
      - Introduces *bottleneck_activation_threshold*: replaces binary sign-gating with soft threshold-based activation
        of bottleneck importance (upward_rank * remaining_work), enabling graceful ramp-up of criticality near deadlines.
      - Retains uncertainty-slack interaction from Parent 2 but applies it to *both* uncertainty and bottleneck terms
        under deadline stress — captures joint risk amplification in constrained regions.
      - Uses median-based normalization (Parent 1) for all features — more robust to outlier-dominated DAGs than mean-abs.
      - Combines exponential wait saturation (Parent 2) with bounded linear duration-uncertainty blend (Parent 1/2).
      - Removes redundant components (e.g., separate comm/exec bias, fairness terms) to reduce noise and improve CMA-ES convergence.
    """
    eps = 0.00026131101539168464
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
        scale = np.median(abs_x) if np.all(np.isfinite(abs_x)) and np.median(abs_x) > eps else eps
        return x / (scale + eps)
    slack_norm = normalize(slk)
    slack_penalty = np.where(slk < 0, 4.3339160484855945 * np.abs(slack_norm), -4.461292175509833 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.8199030479494029 * normalize(inv_energy)
    abs_slack_norm = np.abs(slack_norm)
    bottleneck_gate = np.clip(1.0 - abs_slack_norm / (0.2650642271117642 + eps), 0.0, 1.0)
    bottleneck_importance = rank * work
    bottleneck_active = bottleneck_gate * bottleneck_importance
    bottleneck_score = -1.415751996710781 * normalize(bottleneck_active + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.6828556253428549 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_sat = 1.0 - np.exp(-0.01177737029540912 * wait)
    wait_score = -normalize(wait_sat + eps)
    unc_slack_mask = np.where(slack_norm < 0, np.abs(slack_norm), 0.0)
    unc_slack_interaction = 0.27044738713475414 * (uncert_norm + 0.24989687426606372 * bottleneck_active) * unc_slack_mask
    score = slack_penalty + energy_score + bottleneck_score + dur_score + wait_score + unc_slack_interaction
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 5060740.538709003
    min_safe = -finfo.max / 5060740.538709003
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
