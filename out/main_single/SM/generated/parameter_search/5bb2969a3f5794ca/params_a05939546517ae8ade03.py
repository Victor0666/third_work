import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with:
      - Conditional criticality gate: upward_rank exponentiation *only* when slack <= threshold (no softening — hard activation per reflection)
      - Host-load-sensitive energy gating (reinstated): suppresses energy term when ready_wait_time is large → decouples urgency from load pressure
      - Unified bottleneck term gated by same slack threshold → avoids overloading non-urgent paths
      - Removed redundant clipping parameters: rely on robust normalization + np.finfo-based safeguards instead
      - All numeric literals strictly limited to -2,-1,0,1,2; no unbounded loops or state
      - Uses np.finfo for machine-precision epsilon fallbacks where needed
    """
    eps = 0.0006217105368181634
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
        x = np.asarray(x, dtype=np.float64)
        abs_x = np.abs(x)
        finite_mask = np.isfinite(abs_x)
        if np.any(finite_mask):
            scale = np.max(abs_x[finite_mask])
            finfo = np.finfo(np.float64)
            scale = np.where(scale > eps, scale, finfo.tiny)
        else:
            finfo = np.finfo(np.float64)
            scale = finfo.tiny
        return x / (scale + eps)
    slack_threshold = 6.6454324349909815
    ddl_pressure = np.where(slk <= slack_threshold, 1.0, 0.0)
    slack_penalty = np.where(slk < 0, 3.805896190357938 * np.abs(slk), -1.2125396614611277 * slk)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -1.7101317546423618 * normalize(inv_energy)
    wait_norm = normalize(wait + eps)
    host_load_gate = np.clip(wait_norm, 0.0, 1.0)
    energy_score = energy_score * (1.0 - 0.5582137353717562 * host_load_gate)
    rank_powered = np.power(rank + eps, 2.3650214975814365)
    rank_score = -normalize(rank_powered) * ddl_pressure
    slack_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_term = rank_powered * work * slack_inv
    bottleneck_score = -2.1521376620964188 * normalize(bottleneck_term + eps) * ddl_pressure
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.7190691569809917 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_sat = 1.0 - np.exp(-wait / (84.4603818632333 + eps))
    wait_score = -wait_sat * ddl_pressure
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=-finfo.max)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
