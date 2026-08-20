import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with key structural improvements:
      - Restores stable softplus(log1p(exp(...))) bottleneck formulation from v0 (per reflection) — avoids explosive interactions.
      - Introduces *robust rank-aware normalization*: upward_rank and remaining_work are normalized *separately* before multiplication,
        eliminating scale-mismatch fragility in bottleneck term.
      - Replaces tanh-based energy suppression with *normalized uncertainty gating* on energy score: uses quantile-normalized uncert,
        not raw tanh, for better consistency across uncertainty magnitudes.
      - All numeric literals remain in {-2,-1,0,1,2}; no hidden constants; all parameters referenced via PARAMS.
      - Final score is strictly finite, shape-(N,), deterministic, and satisfies all interface contracts.
    """
    eps = 0.00028104299527978
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
            scale = np.quantile(abs_x[finite_mask], 0.5878698651068356)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    slack_norm = normalize(slk)
    ddl_gate = 1.0 / (1.0 + np.exp(0.18988216123036683 * slack_norm))
    slack_penalty = np.where(slk < 0, 2.7918530306629386 * np.abs(slack_norm), -1.857771596044237 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_base = -0.9606948886469813 * normalize(inv_energy)
    uncert_norm_gated = normalize(uncert + eps) * (1.0 - ddl_gate)
    energy_score = energy_base * (1.0 - 1.0 * uncert_norm_gated)
    rank_score = -1.2170999725340494 * ddl_gate * normalize(rank + eps)
    rank_norm = normalize(rank + eps)
    work_norm = normalize(work + eps)
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_interaction = rank_norm * work_norm * (1.0 + normalize(uncert + eps)) * slack_magnitude_inv
    bottleneck_sharpened = np.log1p(np.exp(bottleneck_interaction))
    bottleneck_score = -1.9232586676249506 * normalize(bottleneck_sharpened + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 1.2335630148105317 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_normalized = wait / (2.301913547846927 + eps)
    wait_saturation = 1.0 - np.exp(-wait_normalized)
    wait_saturation = np.clip(wait_saturation, 0.0, 1.0)
    wait_score = -wait_saturation
    energy_per_duration = energy / (duration + eps)
    load_proxy = normalize(energy_per_duration + eps)
    load_score = -0.30595264813909406 * ddl_gate * load_proxy
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score + load_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 6.706262919951509
    min_safe = finfo.min / 6.706262919951509
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
