import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining strengths of both parents:
      - Max-abs normalization (Parent 2) for stability, augmented with robust energy scaling to balance objective contributions.
      - Dual-gated energy scoring: piecewise-linear DDL gate * exponential slack decay (Parent 2) + host-load sensitivity modulation.
      - Criticality uses exponentiated upward_rank (Parent 2), bottleneck uses rank*work/(|slack|+eps) — monotonic & singularity-free.
      - Duration-uncertainty blend reinstated for fuzzy deadline resilience.
      - Starvation mitigation via exponential wait saturation.
      - Added robust_energy_normalization_scale to prevent energy term from overwhelming slack-driven urgency.
      - All operations guarded against NaN/inf; deterministic; shape-enforced.
    """
    eps = 0.0025798353171469185
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
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    gate_width = 1.8215015711904115 + eps
    ddl_gate = np.where(slk <= 0, 1.0, np.where(slk <= gate_width, 1.0 - slk / gate_width, 0.0))
    slack_penalty = np.where(slk < 0, 2.5551067174572233 * np.abs(slk), -3.6138259215261423 * slk)
    inv_energy = 1.0 / (energy + eps)
    energy_norm = normalize(inv_energy) * 0.9819766117851177
    energy_score = -1.0595910475515196 * energy_norm
    slack_decay = np.exp(-np.abs(slk) * 0.48958009554060866)
    host_load_gate = 1.0 - 0.23637931359097564 * ddl_gate
    energy_score = energy_score * host_load_gate * slack_decay
    rank_powered = np.power(rank + eps, 2.993279634544125)
    rank_score = -normalize(rank_powered)
    slack_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_term = rank * work * slack_inv
    bottleneck_score = -0.38099638414723913 * normalize(bottleneck_term + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.47184976433054376 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_sat = 1.0 - np.exp(-wait / (25.7539987434749 + eps))
    wait_score = -wait_sat
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max / 2.0, neginf=finfo.min / 2.0)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
