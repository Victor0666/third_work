import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule featuring:
      - Smooth sigmoid DDL protection gate (replacing hard threshold) for differentiable, robust transitions.
      - Removal of redundant bottleneck term (rank * work) — avoids duplication and rank instability.
      - Fixed [-2, 2] clipping in normalization (no adaptive saturation) for outlier resilience.
      - Unified load-pressure replaced by leaner, interpretable components: no joint rank×work×uncert coupling.
      - All gating and amplifications now driven by a single smooth sigmoid(gate_center, steepness).
      - Simplified structure: 7 additive terms, 1 conditional branch (sigmoid), zero loops or hidden state.
    """
    eps = 0.012443529510076136
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
    gate_input = 9.509742313252032 * (0.10546618661205963 - slack_norm)
    gate_input_clipped = np.clip(gate_input, -np.log(np.finfo(float).max), np.log(np.finfo(float).max))
    ddl_gate = 1.0 / (1.0 + np.exp(-gate_input_clipped))
    slack_penalty = np.where(slk < 0, 3.893084899030025 * np.abs(slack_norm), -3.3987845445095757 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -2.195726294956859 * normalize(inv_energy)
    rank_score = -2.9895845638939433 * normalize(rank + eps) * ddl_gate
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 1.591928324505722 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_clipped = np.clip(wait, 0.0, 44.309226137681854)
    wait_normalized = wait_clipped / (44.309226137681854 + eps)
    wait_score = -normalize(wait_normalized + eps)
    dur_score = dur_score * (1.0 - ddl_gate * (1.0 - 0.21395070835860333))
    energy_score = energy_score * (1.0 + ddl_gate * (1.0335709617526367 - 1.0))
    rank_score = rank_score * (1.0 + ddl_gate * (1.904369579968003 - 1.0))
    score = slack_penalty + energy_score + rank_score + dur_score + wait_score
    score = np.where(np.isfinite(score), score, 0.0)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
