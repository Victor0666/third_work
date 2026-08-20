import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with soft DDL-protection gating and robust z-score clipping.
    
    Key structural improvements:
      - Replaces hard gating (np.where) with **soft sigmoid-based DDL protection**: 
        criticality and energy terms are smoothly scaled by sigmoid(-slack_norm * scale), 
        enabling graceful degradation under increasing deadline pressure instead of abrupt on/off.
      - Introduces **robust z-score normalization with ±sigma clipping**: computes mean/std only over finite values, 
        then clips outliers beyond ±z_score_clip_sigma * std — more discriminative than median alone, 
        and more stable than raw mean-abs across heterogeneous workloads.
      - Retains bottleneck boost but refines activation: now gated by *both* negative slack *and* non-zero remaining_work, 
        avoiding false bottlenecks from rank-only spikes in low-work leaves.
      - All components remain additive, sign-consistent, and shape-preserving; no loops or hidden state.
    """
    eps = 3.701931479997278e-05
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
        finite_mask = np.isfinite(x) & (np.abs(x) > eps)
        if not np.any(finite_mask):
            return np.zeros_like(x)
        x_finite = x[finite_mask]
        mu = np.mean(x_finite)
        std = np.std(x_finite) if len(x_finite) > 1 else eps
        z = (x - mu) / (std + eps)
        clip_bound = 1.4621110535755202
        z_clipped = np.clip(z, -clip_bound, clip_bound)
        max_abs = np.maximum(np.max(np.abs(z_clipped)), eps)
        return z_clipped / (max_abs + eps)
    slack_norm = normalize(slk)
    ddl_gate = 1.0 / (1.0 + np.exp(2.1465954647235455 * slack_norm))
    slack_penalty = np.where(slk < 0, 7.9069643049841245 * np.abs(slack_norm), -1.2863934986746335 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.5667936927518723 * normalize(inv_energy) * ddl_gate
    rank_score = -1.950063820029183 * normalize(rank) * ddl_gate
    bottleneck_boost = rank * work
    bottleneck_active = np.where((slk < 0) & (work > eps), bottleneck_boost, 0.0)
    bottleneck_score = -1.031726468027679 * normalize(bottleneck_active + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 1.2987848380598466 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_sat = 1.0 - np.exp(-0.12924737152654006 * wait)
    wait_score = -normalize(wait_sat + eps)
    unc_slack_interaction = 1.5641372421823185 * uncert_norm * np.where(slack_norm < 0, np.abs(slack_norm), 0.0)
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score + unc_slack_interaction
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 40421844.955392964
    min_safe = -finfo.max / 40421844.955392964
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
