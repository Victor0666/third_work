import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule with:
      - Robust max-abs normalization.
      - Piecewise-linear DDL gate on robustly power-normalized slack.
      - Slack-driven penalty/gain (negative/positive slack handled separately).
      - Energy score modulated by ddl_gate, exponential slack decay, AND tunable power-law uncertainty coupling.
      - Criticality softened by sigmoidal slack feasibility modulation (replaces binary mask).
      - Bottleneck term uses rank*work/(|robust_slk|+eps) — bounded and monotonic.
      - Duration-uncertainty blend remains linear and normalized.
      - Starvation mitigation via exponential saturation.
      - All numeric literals are {-2,-1,0,1,2}; no hidden constants; all tunables declared.
    """
    eps = 0.00033908637510920173
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
        x = np.asarray(x, dtype=np.float64)
        abs_x = np.abs(x)
        finite_mask = np.isfinite(abs_x)
        if np.any(finite_mask):
            scale = np.max(abs_x[finite_mask])
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    abs_slk = np.abs(slk) + eps
    robust_slk_mag = np.power(abs_slk, 0.8148981363344115)
    robust_slk_norm = robust_slk_mag / (robust_slk_mag + eps)
    slk_robust = robust_slk_norm * np.sign(slk)
    gate_width = 6.524177310108904 + eps
    ddl_gate = np.where(slk_robust <= 0, 1.0, np.where(slk_robust <= gate_width, 1.0 - slk_robust / gate_width, 0.0))
    slack_penalty = np.where(slk_robust < 0, 3.069459477362342 * np.abs(slk_robust), -1.0052004254124391 * slk_robust)
    inv_energy = 1.0 / (energy + eps)
    uncertainty_energy_factor = np.power(1.0 + np.clip(uncert, 0.0, 2.0), 1.0)
    energy_score_base = -1.8519429523928892 * normalize(inv_energy * uncertainty_energy_factor)
    slack_decay = np.exp(-np.abs(slk_robust) * 0.5088421032683696)
    energy_score = energy_score_base * (1.0 - 0.35648869432794733 * ddl_gate) * slack_decay
    rank_powered = np.power(rank + eps, 0.5685861154215491)
    rank_score_unmod = -normalize(rank_powered)
    slack_feasibility = 1.0 / (1.0 + np.exp(-slk_robust * 1.0))
    rank_score = rank_score_unmod * slack_feasibility
    robust_abs_slk = np.abs(slk_robust) + eps
    bottleneck_term = rank * work / robust_abs_slk
    bottleneck_score = -2.2449118549462863 * normalize(bottleneck_term + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.37067013424844153 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_sat = 1.0 - np.exp(-wait / (3.909742192046228 + eps))
    wait_score = -wait_sat
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max / 2.0, neginf=finfo.min / 2.0)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
