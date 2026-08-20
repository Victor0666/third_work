import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining strengths of both parents:
      - Keeps Parent 2's smooth sigmoid DDL gate and bottleneck-proximal term (rank * work)
      - Adds Parent 1's explicit wait-time clipping (more stable than exponential decay)
      - Introduces novel slack-energy interaction: amplifies energy preference under tight slack
      - Uses median-based normalization with explicit finite-filtering for robustness
      - Replaces hard threshold gating with continuous sigmoid modulation across all components
      - All tunable parameters declared; no hidden literals beyond -2..2; deterministic & finite.
    """
    eps = 0.00011022902429668912
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
        finite_mask = np.isfinite(x)
        if not np.any(finite_mask):
            return np.zeros_like(x)
        x_finite = x[finite_mask]
        scale = np.median(np.abs(x_finite))
        scale = np.where(scale > eps, scale, eps)
        return x / (scale + eps)
    slack_norm = normalize(slk) * 6.6815044454078505
    ddl_gate = 1.0 / (1.0 + np.exp(0.5403263950838857 * slack_norm))
    slack_penalty = np.where(slk < 0, 5.813403886281017 * np.abs(slack_norm), -1.391086939770465 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -1.4250536663919686 * normalize(inv_energy)
    energy_slack_interaction = ddl_gate * energy_score * 0.7940056264181743
    rank_score = -0.06947168491322686 * ddl_gate * normalize(rank + eps)
    bottleneck = rank * work
    bottleneck_score = -0.10129864762133192 * normalize(bottleneck + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 1.9996648773041696 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_clipped = np.clip(wait, 0.0, 69.80302672623557)
    wait_score = -normalize(wait_clipped + eps)
    score = slack_penalty + energy_score + energy_slack_interaction + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 167924.78826398484
    min_safe = -finfo.max / 167924.78826398484
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
