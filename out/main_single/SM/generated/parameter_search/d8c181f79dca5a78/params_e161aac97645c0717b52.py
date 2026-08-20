import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule with:
      - Smooth sigmoid ddl_gate + sharpened successor-release term
      - Linear wait saturation (exponent=1, literal allowed)
      - Multiplicative energy-slack coupling via ddl_gate^1
      - Robust quantile normalization throughout
      - All numeric literals restricted to {-2,-1,0,1,2}
      - Exactly 12 parameters
    """
    eps = 0.09997649823886054
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
        finite_mask = np.isfinite(abs_x)
        if np.any(finite_mask):
            scale = np.quantile(abs_x[finite_mask], 0.7769037041212465)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    slack_norm = normalize(slk)
    ddl_gate = 1.0 / (1.0 + np.exp(0.2658577920835636 * slack_norm))
    slack_penalty = np.where(slk < 0, 4.548230501003868 * np.abs(slack_norm), -2.965308324814935 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -1.1924029332323225 * normalize(inv_energy)
    energy_score = energy_score * np.power(ddl_gate, 1)
    rank_score = -1.2527876028705387 * ddl_gate * normalize(rank + eps)
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_sharpened = rank * work * np.power(slack_magnitude_inv, 2.4778147452448707)
    bottleneck_score = -1.0948229262263045 * normalize(bottleneck_sharpened + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 1.773987403888917 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_clipped = np.clip(wait, 0.0, 6.150342064514446)
    wait_normalized = normalize(wait_clipped + eps)
    wait_saturation = np.clip(wait_normalized, 0.0, 1.0)
    wait_score = -wait_saturation
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 697209782.1303316
    min_safe = -finfo.max / 697209782.1303316
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
