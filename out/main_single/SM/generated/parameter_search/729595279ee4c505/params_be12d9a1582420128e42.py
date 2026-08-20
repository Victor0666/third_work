import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with three key structural improvements:
      - Replaces additive host-load term with multiplicative load-bottleneck coupling: 
        `load_proxy × bottleneck_sharpened × ddl_feasible_mask`, ensuring energy-awareness
        is only activated where both bottleneck pressure AND deadline safety coexist.
      - Uses bounded sigmoidal slack-magnitude inversion: `1 / (1 + exp(-k * |slack|))` instead of raw power,
        stabilizing gradients near zero slack and suppressing numerical noise in critical regions.
      - Merges duration & uncertainty *before* quantile normalization via `exec_t + comm_t + uncertainty`,
        reducing feature redundancy and improving signal coherence for temporal risk perception.
      - Removed unused `duration_uncertainty_ratio` parameter to satisfy validation.
    """
    eps = 0.00432955584834875
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    work = np.asarray(remaining_work, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)

    def normalize_abs_quantile(x, q=0.6404710721670082):
        x = np.asarray(x)
        abs_x = np.abs(x)
        finite_mask = np.isfinite(abs_x)
        if np.any(finite_mask):
            scale = np.quantile(abs_x[finite_mask], q)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)

    def normalize_pos_quantile(x, q=0.6404710721670082):
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
    slack_penalty = np.where(slk < 0, 10.640909524482085 * np.abs(slack_norm), -2.4715881319529087 * np.abs(slack_norm))
    margin = 0.2960789182114441
    ddl_feasible_mask = np.clip((slk + margin) / (2 * margin + eps), 0.0, 1.0)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -2.99930225772371 * normalize_pos_quantile(inv_energy) * ddl_feasible_mask
    rank_score = -1.4258164263112598 * normalize_pos_quantile(rank + eps)
    slack_abs = np.abs(slk)
    slack_sigmoid_inv = 1.0 / (1.0 + np.exp(-0.9604834076754382 * (slack_abs - eps)))
    bottleneck_sharpened = rank * work * slack_sigmoid_inv
    bottleneck_score = -0.7988069212530973 * normalize_pos_quantile(bottleneck_sharpened + eps)
    temporal_risk = exec_t + comm_t + uncert
    temporal_risk_norm = normalize_abs_quantile(temporal_risk + eps)
    dur_score = temporal_risk_norm
    wait_clipped = np.clip(wait, 0.0, 8.522071681668482)
    wait_normalized = normalize_pos_quantile(wait_clipped + eps)
    wait_ramp = np.clip(wait_normalized, 0.0, 1.0)
    wait_score = -wait_ramp
    energy_per_duration = energy / (exec_t + comm_t + eps)
    load_proxy = normalize_pos_quantile(energy_per_duration + eps)
    load_bottleneck_interaction = load_proxy * bottleneck_sharpened * ddl_feasible_mask
    load_score = -0.4452360209326153 * normalize_pos_quantile(load_bottleneck_interaction + eps)
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score + load_score
    finfo = np.finfo(np.float64)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max / 2.0, neginf=finfo.min / 2.0)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
