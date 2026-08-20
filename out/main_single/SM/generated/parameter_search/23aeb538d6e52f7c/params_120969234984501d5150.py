import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with:
      - Hard DDL feasibility mask (binary) for strict deadline-first enforcement.
      - Host-load–aware energy gating: uses sublinear (wait + uncertainty)^exponent as proxy for local host congestion.
      - Bottleneck interaction simplified to upward_rank * min(min_exec_time, min_comm_time) — avoids slack singularity and focuses on dominant data/compute bound.
      - All normalizations use robust quantile scaling; no mean/median bias.
      - Strict DDL-first ordering preserved via dominant slack_penalty and hard-gated energy scoring.
    """
    eps = 3.7194044087801197e-06
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
            scale = np.quantile(abs_x[finite_mask], 0.5909527921396313)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    slk_norm = normalize(slk)
    rank_norm = normalize(rank + eps)
    exec_norm = normalize(exec_t + eps)
    comm_norm = normalize(comm_t + eps)
    energy_norm = normalize(energy + eps)
    wait_norm = normalize(wait + eps)
    uncert_norm = normalize(uncert + eps)
    slack_penalty = np.where(slk_norm < 0, 9.293719014648396 * np.abs(slk_norm), -0.10903691326991889 * slk_norm)
    ddl_feasibility_margin = 0.021328312148278464
    ddl_feasible_mask = np.where(slk >= -ddl_feasibility_margin, 1.0, 0.0)
    host_load_proxy = np.power(wait + uncert + eps, 0.9730059385806539)
    host_load_gate = 1.0 / (1.0 + host_load_proxy)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -2.536286571290623 * normalize(inv_energy) * ddl_feasible_mask * host_load_gate
    rank_score = -0.9785316444896008 * rank_norm
    min_duration_bound = np.minimum(exec_t, comm_t)
    bottleneck_interaction = rank_norm * normalize(min_duration_bound + eps)
    bottleneck_score = -0.10299081041092929 * normalize(bottleneck_interaction + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    dur_uncert_blend = dur_norm + 0.5434523928155597 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_saturation = 1.0 - np.exp(-wait / (28.35424999374683 + eps))
    wait_score = -normalize(wait_saturation + eps)
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 6303667725.355967
    min_safe = -finfo.max / 6303667725.355967
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
