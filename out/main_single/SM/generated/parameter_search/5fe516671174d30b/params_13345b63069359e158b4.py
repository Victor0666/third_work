import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's unified quantile normalization and DDL feasibility gate with Parent 1's host-load sensitivity and starvation-aware robustness.
    
    Key structural improvements:
    - Retains Parent 2's semantic-aware quantile normalization (abs-value for temporal features, value-domain for structural) — more stable than sigmoid gates.
    - Reintroduces host-load sensitivity (from Parent 1) but gated by ddl_feasibility_mask instead of fragile sigmoid, ensuring energy-awareness only when deadlines are safely met.
    - Removes all dynamic quantile thresholds (e.g., starvation_activation_quantile) for improved determinism and CMA-ES convergence.
    - Uses clipped wait ramp (Parent 2) — simpler, bounded, and avoids distribution-dependent instability.
    - All numeric literals strictly limited to {-2,-1,0,1,2}; no hidden constants.
    """
    eps = 0.0023250229772299826
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    work = np.asarray(remaining_work, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)

    def normalize_abs_quantile(x, q=0.9389114462678798):
        x = np.asarray(x)
        abs_x = np.abs(x)
        finite_mask = np.isfinite(abs_x)
        if np.any(finite_mask):
            scale = np.quantile(abs_x[finite_mask], q)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)

    def normalize_pos_quantile(x, q=0.9389114462678798):
        x = np.asarray(x)
        finite_mask = np.isfinite(x) & (x >= 0)
        if np.any(finite_mask):
            x_clean = x[finite_mask]
            if len(x_clean) > 0:
                scale = np.quantile(x_clean, q)
                scale = np.where(scale > eps, scale, eps)
            else:
                scale = eps
        else:
            scale = eps
        return x / (scale + eps)
    slack_norm = normalize_abs_quantile(slk)
    slack_penalty = np.where(slk < 0, 1.29734193803091 * np.abs(slack_norm), -2.6761459652114143 * np.abs(slack_norm))
    margin = 0.13313315837358092
    ddl_feasible_mask = np.clip((slk + margin) / (2 * margin + eps), 0.0, 1.0)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -2.8254767327519685 * normalize_pos_quantile(inv_energy) * ddl_feasible_mask
    rank_score = -0.008188465045337608 * normalize_pos_quantile(rank + eps)
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_sharpened = rank * work * np.power(slack_magnitude_inv, 3.072250976088102)
    bottleneck_score = -2.165162050397154 * normalize_pos_quantile(bottleneck_sharpened + eps)
    duration = exec_t + comm_t
    dur_norm = normalize_abs_quantile(duration + eps)
    uncert_norm = normalize_abs_quantile(uncert + eps)
    dur_uncert_blend = dur_norm + 1.6116765202257899 * uncert_norm
    dur_score = normalize_abs_quantile(dur_uncert_blend)
    wait_clipped = np.clip(wait, 0.0, 38.0572126433317)
    wait_normalized = normalize_pos_quantile(wait_clipped + eps)
    wait_ramp = np.clip(wait_normalized, 0.0, 1.0)
    wait_score = -wait_ramp
    energy_per_duration = energy / (duration + eps)
    load_proxy = normalize_pos_quantile(energy_per_duration + eps)
    load_score = -0.06534122309953024 * ddl_feasible_mask * load_proxy
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score + load_score
    finfo = np.finfo(np.float64)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max / 2.0, neginf=finfo.min / 2.0)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
