import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining best practices:
      - Max-abs normalization (Parent 2) for robustness and stability under skew.
      - Piecewise-linear DDL gate (Parent 2) — sharp, interpretable, no fragile sigmoid tuning.
      - Exponential starvation mitigation (Parent 2): 1 - exp(-wait/θ), monotonic and saturating.
      - Novel slack-aware criticality: upward_rank raised to PARAMS["criticality_exponent"] * (1 + |slack|/max_slack)^(-1)
        → softens criticality when slack is large, avoids over-prioritizing non-urgent critical tasks.
      - Enhanced bottleneck term: rank^c * work / (|slack| + eps), normalized — directly couples urgency and workload.
      - Energy suppression gated by ddl_energy_suppression * ddl_gate (not exponentiated, simpler and more interpretable).
      - All parameters used; no numeric literals except -2,-1,0,1,2; strict shape & finiteness enforcement.
    """
    eps = 0.00044373489596929197
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
    slack_abs = np.abs(slk)
    max_slack = np.max(slack_abs) + eps
    ddl_gate = np.where(slk <= 0, 1.0, np.clip(1.0 - slack_abs / max_slack, 0.0, 1.0))
    slack_penalty = np.where(slk < 0, 6.457522349573266 * np.abs(slk), -2.965886250622629 * slk)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -1.4924849091978571 * normalize(inv_energy)
    energy_score = energy_score * (1.0 - 0.5640628143201701 * ddl_gate)
    rel_slack_mag = 1.0 / (1.0 + slack_abs / max_slack)
    rank_powered = np.power(rank + eps, 0.7435149180799527)
    rank_modulated = rank_powered * rel_slack_mag
    rank_score = -normalize(rank_modulated)
    slack_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_term = rank_powered * work * slack_inv
    bottleneck_score = -3.2534556733086086 * normalize(bottleneck_term + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 1.095174282678005 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_sat = 1.0 - np.exp(-wait / (11.635465383566691 + eps))
    wait_score = -wait_sat
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    score_clip_max = 781795.7748875398
    score_clip_min = -899568300.4342645
    score = np.nan_to_num(score, nan=0.0, posinf=score_clip_max, neginf=score_clip_min)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
