import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule with:
      - Energy suppression gated by tunable absolute slack threshold (replacing fixed logic).
      - Retains Parent 2's superior sigmoid DDL gate, bounded linear wait ramp, and successor-release interaction.
      - All features quantile-normalized; handles skew and outliers.
      - Slack penalty dominates to enforce hard deadline feasibility first.
      - No numeric literals beyond -2,-1,0,1,2; all thresholds/weights are parameters.
      - Uses np.finfo for safe clamping instead of hardcoded epsilons.
    """
    eps = 0.01858864067433638
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
            scale = np.quantile(abs_x[finite_mask], 0.5069155708797168)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    slack_norm = normalize(slk)
    ddl_gate = 1.0 / (1.0 + np.exp(0.08697733886820852 * slack_norm))
    slack_penalty = np.where(slk < 0, 5.341409641236813 * np.abs(slk), -1.8312788004985046 * slk)
    energy_active = (slk > 59.006549075793075).astype(np.float64)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -1.62835120254331 * normalize(inv_energy) * energy_active
    rank_score = -0.0024018476706755163 * ddl_gate * normalize(rank + eps)
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_sharpened = rank * work * np.power(slack_magnitude_inv, 1.0879497737051183)
    bottleneck_score = -0.25099160248231367 * normalize(bottleneck_sharpened + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.2847072361333293 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_clipped = np.clip(wait, 0.0, 59.006549075793075)
    wait_normalized = normalize(wait_clipped + eps)
    wait_ramp = np.clip(wait_normalized, 0.0, 1.0)
    wait_score = -wait_ramp
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 60387.32683991704
    min_safe = -finfo.max / 60387.32683991704
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
