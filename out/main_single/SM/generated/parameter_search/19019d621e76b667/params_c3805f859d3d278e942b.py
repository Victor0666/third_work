import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule with three key structural improvements:
      - Replaced sigmoid DDL gate with piecewise-linear feasibility margin: explicit 'safe zone' where energy dominates,
        improving interpretability and eliminating exponential sensitivity near zero.
      - Introduced conditional energy modulation: energy term is scaled by a smooth transition from full suppression
        (under deadline pressure) to full activation (in safe zone), replacing hard gating.
      - Removed finfo_max_scale (inactive per diagnostics) and simplified NaN/inf handling using np.clip + eps-based bounds.
      - All normalizations use adaptive quantile scaling; no median or mean bias.
      - Strict DDL-first ordering preserved via dominant slack_penalty term and piecewise feasibility logic.
    """
    eps = 0.003901230733645387
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
            scale = np.quantile(abs_x[finite_mask], 0.6231872345547866)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    slack_norm = normalize(slk)
    margin = 0.7467514556596903
    ddl_feasible_mask = np.clip((slk + margin) / (2 * margin + eps), 0.0, 1.0)
    slack_penalty = np.where(slk < 0, 3.2442273104192267 * np.abs(slack_norm), -0.1373676522091753 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.20585629808476674 * normalize(inv_energy) * ddl_feasible_mask
    rank_score = -0.2434851024286966 * normalize(rank + eps)
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_sharpened = rank * work * np.power(slack_magnitude_inv, 1.4369923772841153)
    bottleneck_score = -0.6929560256082219 * normalize(bottleneck_sharpened + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.23962870785703652 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_clipped = np.clip(wait, 0.0, 34.47071781930879)
    wait_normalized = normalize(wait_clipped + eps)
    wait_ramp = np.clip(wait_normalized, 0.0, 1.0)
    wait_score = -wait_ramp
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    score = np.nan_to_num(score, nan=0.0, posinf=201016949780.04517, neginf=-201016949780.04517)
    score = np.clip(score, -201016949780.04517, 201016949780.04517)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
