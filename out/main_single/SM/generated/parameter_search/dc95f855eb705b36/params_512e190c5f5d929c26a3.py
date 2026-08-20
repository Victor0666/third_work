import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule with smooth DDL protection gate, successor-release interaction, and host-load-aware energy scaling.
    
    Key structural mutations:
      - Replaces hard `slack_norm >= 0` gating with *smooth sigmoid gate* centered at slack=0 (controlled by ddl_protection_gate_width)
        → eliminates discontinuities, improves gradient stability for CMA-ES, and enables graceful degradation under marginal slack.
      - Adds *successor-release interaction*: prioritizes tasks whose completion releases many successors *only when system load is high* 
        (estimated via median-normalized remaining_work), addressing 'critical_path starvation' observed in replay failures.
      - Introduces *host-load-aware energy scaling*: applies power-law compression to energy_score when total remaining_work is high,
        preventing over-aggressive low-energy selection that delays critical path tasks under congestion.
      - Uses *median-based robust normalization* (instead of mean) for all feature scalings to suppress outlier influence.
      - All divisions guarded; all NaN/inf replaced; shape strictly enforced; deterministic.
    """
    eps = 0.0027095178009985924
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
        med_abs = np.median(abs_x)
        scale = med_abs if np.all(np.isfinite(abs_x)) and med_abs > eps else eps
        return x / (scale + eps)
    slack_centered = slk / (np.median(np.abs(slk)) + eps)
    gate_smooth = 1.0 / (1.0 + np.exp(-slack_centered / 0.012100839783300255))
    slack_penalty = 6.382373158384944 * np.maximum(0.0, -slack_centered) - 2.534004804526273 * np.maximum(0.0, slack_centered)
    inv_energy = 1.0 / (energy + eps)
    energy_base = -0.3802527323525947 * normalize(inv_energy)
    load_factor = np.clip(normalize(work), 0.0, 2.0)
    energy_score = energy_base * (1.0 - load_factor ** 0.8482116304725147)
    rank_score = -1.1975519990704935 * normalize(rank + eps) * gate_smooth
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.9330229176283023 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_sat = 1.0 - np.exp(-0.14030460424201288 * wait)
    wait_score = -normalize(wait_sat + eps)
    unc_slack_interaction = 3.250038269073956 * uncert_norm * np.maximum(0.0, -slack_centered)
    successor_release_boost = 0.9923459584725947 * normalize(rank + eps) * (1.0 - gate_smooth) * load_factor
    score = slack_penalty + energy_score + rank_score + dur_score + wait_score + unc_slack_interaction + successor_release_boost
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 1188816.447531289
    min_safe = -finfo.max / 1188816.447531289
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
