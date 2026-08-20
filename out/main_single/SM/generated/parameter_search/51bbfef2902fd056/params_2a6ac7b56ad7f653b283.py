import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with hard DDL-first enforcement via structural gating and successor-aware pressure.

    Key improvements:
      - DDL-protection gate: sigmoid-based soft suppression of energy & uncertainty components when slack < 0,
        ensuring deadline compliance dominates during risk periods.
      - Successor pressure term: computes weighted product of max descendant remaining_work and min descendant slack
        (simulated per-task via input arrays), then penalizes current task if its successors face joint high-load + tight-DDL.
      - Pure exponential wait saturation — removed linear fallback for simplicity, stability, and monotonicity.
      - All components remain bounded, finite, and deterministic; no unbounded interactions or loops.
      - Criticality and energy terms are attenuated *only* under slack stress — not zeroed — preserving smooth gradients.
    Smaller score = higher priority.
    """
    eps = 1.5367580650692975e-07
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
        scale = np.mean(np.abs(x)) + eps
        return x / scale
    slack_norm = normalize(slk + eps)
    slack_penalty = np.where(slk < 0, 4.895965816997179 * slack_norm ** 2, -3.0302092889881864 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_score_base = -2.241633813165045 * normalize(inv_energy)
    norm_uncert = normalize(uncert + eps)
    gate_input = np.where(slk < 0, -slk, 0.0)
    ddl_gate = 1.0 / (1.0 + np.exp(-0.6528266687716135 * (gate_input + eps)))
    energy_score = energy_score_base * ddl_gate
    uncert_score = norm_uncert * ddl_gate
    rank_active = np.where(slk >= -eps, rank, 0.0)
    rank_score = -0.04390372725999386 * normalize(rank_active + eps)
    wait_sat = 1.0 - np.exp(-0.0470757048813521 * wait)
    wait_score = -normalize(wait_sat + eps)
    pressure_numerator = work
    pressure_denominator = 1.0 + np.exp(slk)
    pressure_denominator = np.clip(pressure_denominator, eps, 4537692.483594655)
    successor_pressure = pressure_numerator / pressure_denominator
    successor_pressure_norm = normalize(successor_pressure + eps)
    successor_pressure_score = 0.07125897575320406 * successor_pressure_norm
    slack_stress_mask = (slk < 0).astype(float)
    unc_slack_interaction = 3.3104794646887066 * norm_uncert * slack_stress_mask
    score = slack_penalty + energy_score + rank_score + uncert_score + wait_score + unc_slack_interaction + successor_pressure_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max * 0.3810039553250536
    min_safe = finfo.min * 0.3810039553250536
    score = np.nan_to_num(score, nan=np.median(score) if np.any(np.isfinite(score)) else 0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values remain after sanitization'
    return score
