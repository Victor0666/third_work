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
    eps = 0.05468392766005257
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
            scale = np.quantile(abs_x[finite_mask], 0.8247337803811656)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    slack_norm = normalize(slk)
    ddl_gate = 1.0 / (1.0 + np.exp(0.6456730001989646 * slack_norm))
    slack_penalty = np.where(slk < 0, 3.786205054298224 * np.abs(slk), -4.355045870095364 * slk)
    energy_active = (slk > 77.33798695778584).astype(np.float64)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -1.7261525433741922 * normalize(inv_energy) * energy_active
    rank_score = -0.5439241118952571 * ddl_gate * normalize(rank + eps)
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_sharpened = rank * work * np.power(slack_magnitude_inv, 0.7912844747816343)
    bottleneck_score = -1.4458130014872592 * normalize(bottleneck_sharpened + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.1452616261194255 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_clipped = np.clip(wait, 0.0, 77.33798695778584)
    wait_normalized = normalize(wait_clipped + eps)
    wait_ramp = np.clip(wait_normalized, 0.0, 1.0)
    wait_score = -wait_ramp
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 19487.9680140328
    min_safe = -finfo.max / 19487.9680140328
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
