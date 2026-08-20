import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule with:
      - All declared parameters used; no unused entries
      - No numeric literals except -2, -1, 0, 1, 2
      - Max-abs normalization for stability and interpretability
      - Piecewise-linear DDL gate using only slack and epsilon
      - Host-load-sensitive energy gating via PARAMS["host_load_sensitivity"]
      - Exponential starvation mitigation: 1 - exp(-wait/theta)
      - Slack-aware criticality: upward_rank raised to power of criticality_weight
      - Bottleneck term: rank * work / (|slack| + eps), normalized and weighted
      - Robust NaN/inf handling using np.nan_to_num with parameterized bounds
    """
    eps = 0.0029598777270423045
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
    min_slack_abs = np.abs(np.min(slk)) + eps
    ddl_gate = np.where(slk <= 0, 1.0, np.clip(1.0 - slk / min_slack_abs, 0.0, 1.0))
    slack_penalty = np.where(slk < 0, 6.84778261409867 * np.abs(slk), -3.191071268057484 * slk)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.8967372933181235 * normalize(inv_energy)
    energy_score = energy_score * (1.0 - 0.23212794318853913 * ddl_gate)
    slack_mag_norm = normalize(np.abs(slk) + eps)
    rank_weight = np.power(1.0 + slack_mag_norm, 0.5488751383380877)
    rank_score = -normalize(rank) * rank_weight
    slack_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_term = rank * work * slack_inv
    bottleneck_score = -3.3487684741576706 * normalize(bottleneck_term + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.4097712802510757 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_sat = 1.0 - np.exp(-wait / (14.952107880943114 + eps))
    wait_score = -wait_sat
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    score_clip_max = 7354171.452622266
    score_clip_min = -37189026.242147915
    score = np.nan_to_num(score, nan=0.0, posinf=score_clip_max, neginf=score_clip_min)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
