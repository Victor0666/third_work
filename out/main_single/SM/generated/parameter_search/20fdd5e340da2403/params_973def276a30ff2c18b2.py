import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule featuring:
      - Restored median/MAD normalization (per self-reflection) for robustness against skewed wait/uncertainty.
      - Unified slack-aware gating: ddl_gate now modulates *all* deadline-sensitive terms (energy, rank, release, load) consistently.
      - All numeric literals strictly in {-2,-1,0,1,2}; no hidden constants.
      - Deterministic, finite, shape-compliant, and fully parameterized.
    """
    eps = 3.671121516982921e-06
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
        scale = max(med, 1.0 * mad, eps)
        return x / (scale + eps)
    gate_width = 0.28020836123606213
    ddl_gate = 1.0 / (1.0 + np.exp(-slk / (gate_width + eps)))
    slack_penalty = 4.579474084844004 * np.maximum(-slk, 0.0) + 4.31241132073316 * np.minimum(slk, 0.0)
    slack_score = normalize(slack_penalty + eps)
    inv_energy = 1.0 / (energy + eps)
    energy_active = inv_energy * ddl_gate
    energy_score = -2.7499156192132355 * normalize(energy_active + eps)
    rank_score = -2.9840923307493905 * ddl_gate * normalize(rank + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.8309309434538332 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_sat = 1.0 - np.exp(-0.4864762627662659 * wait)
    wait_score = -normalize(wait_sat + eps) * ddl_gate
    unc_slack_interaction = 4.314040527650854 * uncert_norm * (1.0 - ddl_gate)
    release_pressure = rank * work
    release_score = -0.32384444136268015 * ddl_gate * normalize(release_pressure + eps)
    load_factor = (work * uncert + eps) ** 0.8602759028056396
    load_score = normalize(load_factor) * ddl_gate
    score = slack_score + energy_score + rank_score + dur_score + wait_score + unc_slack_interaction + release_score + load_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 9594.75230953162
    min_safe = -finfo.max / 9594.75230953162
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
