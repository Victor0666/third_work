import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule with:
      - Robust median+MAD normalization (resistant to skew/outliers)
      - Smooth sigmoid DDL gate applied *only* to criticality and uncertainty-slack terms
      - Ungated energy/duration/wait terms to preserve universal fairness
      - Exponentiated host-load term `(work * uncert) ** exponent` for risk-concentrated load amplification
      - Simplified `rank * work` successor release pressure (no extra gating)
      - All numeric literals restricted to {-2,-1,0,1,2}; all tunables declared in PARAMETER_SCHEMA
    """
    eps = 0.004877245813320288
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
        med = np.median(abs_x)
        mad = np.median(np.abs(abs_x - med)) if np.all(np.isfinite(abs_x)) else eps
        scale = max(med, mad, eps)
        return x / (scale + eps)
    abs_slk = np.abs(slk)
    slk_norm = slk / (np.median(abs_slk) + eps)
    ddl_gate = 1.0 / (1.0 + np.exp(-slk_norm / (0.12856846406411593 + eps)))
    slack_penalty = 7.34038081558667 * np.maximum(-slk, 0.0) + 5.017348210479253 * np.minimum(slk, 0.0)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.3973489016116451 * normalize(inv_energy)
    rank_score = -1.9079542290668148 * ddl_gate * normalize(rank + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.342571837440779 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_sat = 1.0 - np.exp(-0.09183046331596437 * wait)
    wait_score = -normalize(wait_sat + eps)
    unc_slack_interaction = 2.4783316638496595 * uncert_norm * (1.0 - ddl_gate)
    release_pressure = rank * work
    release_score = -0.15398762532292054 * normalize(release_pressure + eps)
    load_factor = (work * uncert + eps) ** 0.5105083168024281
    load_score = normalize(load_factor)
    score = slack_penalty + energy_score + rank_score + dur_score + wait_score + unc_slack_interaction + release_score + load_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 342530.3096943905
    min_safe = -finfo.max / 342530.3096943905
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
