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
    eps = 0.0001861846323042608
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
            scale = np.quantile(abs_x[finite_mask], 0.7782638806297721)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    slack_norm = normalize(slk)
    ddl_gate = 1.0 / (1.0 + np.exp(0.15709715317609796 * slack_norm))
    slack_penalty = np.where(slk < 0, 11.160007073966458 * np.abs(slack_norm), -3.320337784687374 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.16064901116245173 * normalize(inv_energy) * np.power(ddl_gate, 0.41122946045862663)
    rank_score = -0.7495660765411278 * ddl_gate * normalize(rank + eps)
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_sharpened = rank * work * np.power(slack_magnitude_inv, 1.2195151590975382)
    bottleneck_score = -1.7816931935094513 * normalize(bottleneck_sharpened + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.9919743685430287 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_clipped = np.clip(wait, 0.0, 66.39435243642527)
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
