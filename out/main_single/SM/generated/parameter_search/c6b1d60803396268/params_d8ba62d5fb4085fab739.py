import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule with two structural innovations:
      - Upward rank raised to tunable power to avoid saturation on long critical paths.
      - Energy term gated by |slack|^exponent to suppress under deadline pressure without sigmoid instability.
      - Removed unused ddl_protection_gate parameter and associated logic.
      - All normalizations use robust quantile scaling; no mean/median bias.
      - Slack penalty remains dominant to enforce DDL feasibility first.
      - Wait ramp preserved as bounded linear unit-scale monotonic mitigation.
    """
    eps = 5.921991639728829e-06
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
            scale = np.quantile(abs_x[finite_mask], 0.5780781164980879)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    slack_penalty = np.where(slk < 0, 10.422434517013732 * np.abs(normalize(slk)), -2.6701247513846784 * np.abs(normalize(slk)))
    inv_energy = 1.0 / (energy + eps)
    energy_norm = normalize(inv_energy)
    slack_abs = np.abs(slk) + eps
    energy_gate = np.power(slack_abs, -0.5463624299269553)
    energy_score = -2.9136819903530733 * energy_norm * energy_gate
    rank_pow = np.power(rank + eps, 0.5463624299269553)
    rank_score = -1.8678015477639238 * normalize(rank_pow)
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_sharpened = rank_pow * work * np.power(slack_magnitude_inv, 2.3011933118608807)
    bottleneck_score = -3.258350697742185 * normalize(bottleneck_sharpened + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.7802057894766707 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_clipped = np.clip(wait, 0.0, 16.101119239386023)
    wait_normalized = normalize(wait_clipped + eps)
    wait_ramp = np.clip(wait_normalized, 0.0, 1.0)
    wait_score = -wait_ramp
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
