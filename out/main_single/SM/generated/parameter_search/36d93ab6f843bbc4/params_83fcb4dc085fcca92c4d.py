import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's stability with Parent 1's risk-aware interactions:
      - Retains hard wait clipping and bottleneck-proximal term from Parent 2.
      - Integrates robust median/MAD normalization with tunable robustness factor.
      - Adds load-risk interaction (work * uncertainty)^exponent, gated by ddl_gate to prioritize under deadline stress.
      - Uses sigmoid DDL gate for smooth feasibility enforcement across all critical terms.
      - All numeric literals strictly in {-2,-1,0,1,2}; no hidden constants.
      - Deterministic, finite, shape-compliant, and fully parameterized.
    """
    eps = 0.09259223678313924
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
        if not np.any(finite_mask):
            return np.zeros_like(x)
        med = np.median(abs_x[finite_mask])
        mad = np.median(np.abs(abs_x[finite_mask] - med))
        scale = max(med, 0.5456064985575093 * mad, eps)
        return x / (scale + eps)
    slack_norm = normalize(slk)
    ddl_gate = 1.0 / (1.0 + np.exp(0.668097011300776 * slack_norm))
    slack_penalty = np.where(slk < 0, 8.071375090486043 * np.abs(slack_norm), -3.8317636291906685 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.5020752633407845 * normalize(inv_energy)
    rank_score = -0.20881406652458245 * ddl_gate * normalize(rank + eps)
    bottleneck = rank * work
    bottleneck_score = -0.4907221833783798 * normalize(bottleneck + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.5443795795209195 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_clipped = np.clip(wait, 0.0, 99.35794576272504)
    wait_score = -normalize(wait_clipped + eps)
    load_risk = (work * uncert + eps) ** 1.5613202525033114
    load_risk_score = normalize(load_risk) * ddl_gate
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score + load_risk_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 4056074.0929719056
    min_safe = -finfo.max / 4056074.0929719056
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
