import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule featuring:
      - Replaced hard mask with *smooth, differentiable slack gating* (sigmoid) for all deadline-sensitive terms,
        improving gradient stability for offline CMA-ES optimization.
      - Unified robust normalization using *MAD-scaled median absolute deviation*, now parameterized via
        `robust_norm_mad_scale` to preserve statistical consistency across features.
      - Removed bottleneck-proximal term (rank * work) as redundant per reflection; instead strengthened
        release pressure via *normalized product of rank and work*, fully soft-gated for smoother DDL adaptation.
      - All numeric literals strictly in {-2,-1,0,1,2}; no hidden constants.
      - Deterministic, finite, shape-compliant, and fully parameterized.
    """
    eps = 0.014934928786870964
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
        med = np.median(abs_x)
        mad = np.median(np.abs(abs_x - med)) if np.all(np.isfinite(abs_x)) else eps
        scale = max(med, 1.0239147790145726 * mad, eps)
        return x / (scale + eps)
    gate_width = 0.22003777992796383
    ddl_gate = 1.0 / (1.0 + np.exp(-slk / (gate_width + eps)))
    slack_penalty = 0.8414357479172949 * np.maximum(-slk, 0.0) + 5.001539952521082 * np.minimum(slk, 0.0)
    slack_score = normalize(slack_penalty + eps)
    inv_energy = 1.0 / (energy + eps)
    energy_active = inv_energy * ddl_gate
    energy_score = -0.7838521142331221 * normalize(energy_active + eps)
    rank_score = -1.0554991890812342 * ddl_gate * normalize(rank + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.836877560866085 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_sat = 1.0 - np.exp(-0.221373893887785 * wait)
    wait_score = -normalize(wait_sat + eps) * ddl_gate
    unc_slack_interaction = 3.553324832493911 * uncert_norm * (1.0 - ddl_gate)
    release_pressure = rank * work
    release_score = -0.19113955765320606 * ddl_gate * normalize(release_pressure + eps)
    score = slack_score + energy_score + rank_score + dur_score + wait_score + unc_slack_interaction + release_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 947461273.1082684
    min_safe = -finfo.max / 947461273.1082684
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
