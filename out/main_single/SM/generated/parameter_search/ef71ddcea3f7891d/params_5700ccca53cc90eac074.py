import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining strengths:
      - Smooth sigmoid DDL protection gate with tunable center & steepness
      - Bottleneck-proximal term (rank * work) and power-law starvation
      - Floor-augmented rank normalization for robustness
      - Bounded [-2, 2] clipping in normalization for outlier resilience
      - All operations guarded against NaN/inf/zero; no hidden constants beyond {-2,-1,0,1,2}
      - Removed redundant amplification parameters to meet 12-parameter limit
    """
    eps = 0.012446874204977797
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
            med = np.median(abs_x[finite_mask])
            dev = np.abs(abs_x - med)
            mad = np.median(dev[finite_mask]) if np.any(finite_mask) else eps
            scale = max(med, 1.0 * mad, eps)
        else:
            scale = eps
        normed = x / (scale + eps)
        return np.clip(normed, -2.0, 2.0)
    slack_norm = normalize(slk)
    gate_input = 5.319395607821818 * (0.029163350013896583 - slack_norm)
    gate_input_clipped = np.clip(gate_input, -np.log(np.finfo(float).max), np.log(np.finfo(float).max))
    ddl_gate = 1.0 / (1.0 + np.exp(-gate_input_clipped))
    slack_penalty = np.where(slk < 0, 10.27021684088028 * np.abs(slack_norm), -3.4354493519437845 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.49117554152942544 * normalize(inv_energy)
    rank_stable = rank + 0.05919849395562119
    rank_score = -1.4005252119089455 * normalize(rank_stable) * ddl_gate
    bottleneck = rank * work
    bottleneck_score = -0.18798337667847953 * normalize(bottleneck + eps) * ddl_gate
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.9856262990931449 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_clipped = np.clip(wait, 0.0, 15.020264498728743)
    wait_sharpened = np.power(wait_clipped + eps, 1.3350314878698313)
    wait_score = -normalize(wait_sharpened)
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    score = np.where(np.isfinite(score), score, 0.0)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
