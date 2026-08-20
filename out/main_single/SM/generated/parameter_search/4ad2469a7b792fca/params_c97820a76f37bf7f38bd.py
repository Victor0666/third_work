import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule with three structural improvements:
      - Smooth sigmoid DDL protection gate replaces hard sign-based gating → eliminates discontinuity at slack=0.
      - Successor-release interaction: weighted remaining_work * urgency signal promotes unblocking critical paths.
      - Host-load-aware energy modulation: activates only under high combined uncertainty+slack stress, preventing over-correction.
    All normalizations use median absolute deviation (MAD) scaling for robustness against outliers.
    """
    eps = 0.03582588183762395
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
        mad = np.median(np.abs(abs_x - med)) if np.all(np.isfinite(abs_x)) and med > eps else eps
        scale = mad + eps
        return x / scale
    gate_center = 0.0
    gate_width = 0.17055601100164552
    ddl_gate = 1.0 / (1.0 + np.exp(-(slk - gate_center) / (gate_width + eps)))
    slack_penalty_base = np.where(slk < 0, 1.276380289952591 * np.abs(slk), -3.658334243274249 * np.abs(slk))
    slack_penalty = slack_penalty_base * ddl_gate
    inv_energy = 1.0 / (energy + eps)
    energy_norm = normalize(inv_energy)
    stress_signal = np.abs(slk) * uncert
    stress_normalized = normalize(stress_signal)
    host_load_mod = np.where(stress_normalized > 1.0472831279878347, 1.0 + stress_normalized, 1.0)
    energy_score = -0.8710656567688186 * energy_norm * host_load_mod
    rank_score = -1.9166683461130765 * normalize(rank + eps) * ddl_gate
    urgency_factor = np.clip(-slk, 0.0, np.inf)
    successor_release = 0.7458598627829297 * normalize(work + eps) * normalize(urgency_factor + eps)
    successor_score = -successor_release
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.9396732021849227 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_sat = 1.0 - np.exp(-0.06790473590352587 * wait)
    wait_score = -normalize(wait_sat + eps)
    unc_slack_interaction = 0.1148895503195978 * uncert_norm * (1.0 - ddl_gate)
    score = slack_penalty + energy_score + rank_score + successor_score + dur_score + wait_score + unc_slack_interaction
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 15391074.5796399
    min_safe = -finfo.max / 15391074.5796399
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
