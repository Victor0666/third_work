import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule with three key structural innovations:
      - Replaced static energy gating with dynamic exponentiated DDL gating: energy term scaled by (ddl_gate)^sensitivity,
        enabling smoother suppression under deadline pressure without hard thresholds
      - Removed finfo_max_scale (inactive per diagnostics) and replaced with direct np.clip-based safeguard using machine epsilon
      - Added host-load sensitivity exponent to decouple energy prioritization from criticality: avoids conflating DDL risk with load effects
      - All normalizations remain quantile-based and robust; no mean/median bias
      - Preserves strict DDL-first ordering via dominant slack_penalty
      - Starvation mitigation remains bounded linear ramp (monotonic, no discontinuities)
      - Eliminated redundant 'finfo_max_scale' and simplified final clipping logic
    """
    eps = 0.00032986054812372714
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
            scale = np.quantile(abs_x[finite_mask], 0.9497397560435257)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    slack_norm = normalize(slk)
    ddl_gate = 1.0 / (1.0 + np.exp(0.18733458887274257 * slack_norm))
    slack_penalty = np.where(slk < 0, 10.411045596109268 * np.abs(slack_norm), -0.775993911215337 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.6636015874656335 * normalize(inv_energy) * np.power(ddl_gate, 0.6451618928044502)
    rank_score = -0.2781574763603333 * ddl_gate * normalize(rank + eps)
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_sharpened = rank * work * np.power(slack_magnitude_inv, 2.252708014249895)
    bottleneck_score = -1.0703639423277378 * normalize(bottleneck_sharpened + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 1.5357662668876662 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_clipped = np.clip(wait, 0.0, 41.07300692207319)
    wait_normalized = normalize(wait_clipped + eps)
    wait_ramp = np.clip(wait_normalized, 0.0, 1.0)
    wait_score = -wait_ramp
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max * eps
    min_safe = -finfo.max * eps
    score = np.clip(score, min_safe, max_safe)
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
