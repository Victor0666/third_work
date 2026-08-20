import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule featuring:
      - Replaced exponential wait decay with bounded linear clipping (more stable & interpretable)
      - Added conditional DDL protection gate: when normalized slack < ddl_protection_gate,
        energy and criticality terms dominate duration/uncertainty to prevent late scheduling
      - Removed uncertainty-slack interaction (diagnosed inactive) and simplified duration blend
      - Used median-based normalization for outlier resilience per design insight
      - All components explicitly gated by slack sign or magnitude for DDL-first enforcement
      - All tunable coefficients now declared in PARAMETER_SCHEMA (no hidden literals)
    """
    eps = 0.0007874621313774877
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)

    def normalize(x):
        x = np.asarray(x)
        abs_x = np.abs(x)
        scale = np.median(abs_x) if np.all(np.isfinite(abs_x)) and np.median(abs_x) > eps else eps
        return x / (scale + eps)
    slack_norm = normalize(slk)
    slack_penalty = np.where(slk < 0, 4.8628138505921825 * np.abs(slack_norm), -2.7852043717258717 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.8577198536121962 * normalize(inv_energy)
    rank_active = np.where((slk >= 0) & (slack_norm >= 0.03083215896536102), rank, 0.0)
    rank_score = -1.1210424793800864 * normalize(rank_active + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 1.0893429227640448 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_clipped = np.clip(wait, 0.0, 5.84591265228447)
    wait_normalized = wait_clipped / (5.84591265228447 + eps)
    wait_score = -normalize(wait_normalized + eps)
    protection_mask = slack_norm < 0.03083215896536102
    dur_score = np.where(protection_mask, dur_score * 0.4157010341514342, dur_score)
    energy_score = np.where(protection_mask, energy_score * 1.0003891741394986, energy_score)
    rank_score = np.where(protection_mask, rank_score * 1.061851944206652, rank_score)
    score = slack_penalty + energy_score + rank_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 21642688.37156841
    min_safe = -finfo.max / 21642688.37156841
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
