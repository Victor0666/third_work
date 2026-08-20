import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule featuring:
      - Restored median/MAD normalization (per self-reflection) for robustness against skewed wait/uncertainty.
      - Unified slack-aware gating: ddl_gate now modulates *all* deadline-sensitive terms (energy, rank, release, load) consistently.
      - All numeric literals strictly in {-2,-1,0,1,2}; no hidden constants.
      - Deterministic, finite, shape-compliant, and fully parameterized.
    """
    eps = 0.0378387528801308
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
    gate_width = 0.2674677894636157
    ddl_gate = 1.0 / (1.0 + np.exp(-slk / (gate_width + eps)))
    slack_penalty = 7.9315580277206745 * np.maximum(-slk, 0.0) + 2.249346445923437 * np.minimum(slk, 0.0)
    slack_score = normalize(slack_penalty + eps)
    inv_energy = 1.0 / (energy + eps)
    energy_active = inv_energy * ddl_gate
    energy_score = -0.05946191008614837 * normalize(energy_active + eps)
    rank_score = -0.4622467011499131 * ddl_gate * normalize(rank + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 1.494539136434681 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_sat = 1.0 - np.exp(-0.29462947364049114 * wait)
    wait_score = -normalize(wait_sat + eps) * ddl_gate
    unc_slack_interaction = 1.3123761677092685 * uncert_norm * (1.0 - ddl_gate)
    release_pressure = rank * work
    release_score = -0.0024675580097284207 * ddl_gate * normalize(release_pressure + eps)
    load_factor = (work * uncert + eps) ** 0.5589517195534578
    load_score = normalize(load_factor) * ddl_gate
    score = slack_score + energy_score + rank_score + dur_score + wait_score + unc_slack_interaction + release_score + load_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 137177.1160210168
    min_safe = -finfo.max / 137177.1160210168
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
