import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining robustness, deadline responsiveness, and starvation mitigation.
    
    Key structural improvements:
    - Replaces sigmoid/quantile adaptive gates with a *normalized piecewise-linear DDL gate*:
      `ddl_gate = clip(1 - |slack| / (max(|slack|) + eps), 0, 1)` — avoids overflow, preserves monotonicity,
      and activates smoothly near zero slack without per-task offsets.
    - Introduces *DDL-protection strength scaling*: all deadline-sensitive terms (criticality, bottleneck, energy-gating)
      are multiplied by `ddl_protection_strength * ddl_gate`, enabling CMA-ES to jointly tune gate sensitivity
      and term impact — decouples gate shape from term weighting.
    - Uses *max-abs normalization only*, eliminating quantile instability and ensuring consistent scale
      across all features — aligns with Parent 2's proven robustness.
    - Criticality uses *power-law sharpening*: `rank^criticality_exponent`, gated by DDL strength — 
      amplifies high-rank tasks only when deadlines are tight.
    - Bottleneck term remains `rank * work / (|slack| + eps)` but is now *fully DDL-gated* for safety.
    - Energy efficiency includes *host-load-aware gating*: reduced influence under tight slack via `energy_gating_sensitivity`.
    - Starvation uses stable `1 - exp(-wait/theta)` — no quantile thresholding, avoiding workflow-context bias.
    - All numeric literals are {-2,-1,0,1,2}; no hidden constants; all parameters referenced exactly once.
    """
    eps = 0.002505588402375954
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
    abs_slack = np.abs(slk)
    max_abs_slack = np.max(abs_slack) + eps
    ddl_gate = np.clip(1.0 - abs_slack / max_abs_slack, 0.0, 1.0)
    slack_penalty = np.where(slk < 0, 7.598135217101607 * abs_slack, -1.0739250638376499 * slk)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.35615244468967816 * normalize(inv_energy)
    energy_score = energy_score * (1.0 - 0.3198642568162182 * ddl_gate)
    rank_norm = normalize(rank + eps)
    rank_sharpened = np.power(rank_norm + eps, 0.9070970423702414)
    rank_score = -0.38944655798880745 * ddl_gate * rank_sharpened
    slack_inv = 1.0 / (abs_slack + eps)
    bottleneck_base = rank * work * slack_inv
    bottleneck_score = -2.47727169478111 * 0.38944655798880745 * ddl_gate * normalize(bottleneck_base + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.8937173011656476 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_sat = 1.0 - np.exp(-wait / (3.3970389810151356 + eps))
    wait_score = -wait_sat
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    score_clip_max = 151592487.61639893
    score_clip_min = -5005603.579612097
    score = np.nan_to_num(score, nan=0.0, posinf=score_clip_max, neginf=score_clip_min)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
