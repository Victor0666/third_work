import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule with:
      - Smooth sigmoid DDL protection gate (replaces hard sign gating) for robust slack transition;
      - Load-aware successor release term: rewards tasks that unlock high-criticality descendants;
      - Host-load-sensitive conditional gating: applies energy/criticality penalties only when system load is high;
      - Median-based normalization (more robust than mean under skew);
      - All components explicitly bounded and finite.
    """
    eps = 2.9295551533008124e-05
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
        scale = med + mad + eps
        return x / scale
    slack_centered = slk / (np.median(np.abs(slk)) + eps)
    ddl_gate = 1.0 / (1.0 + np.exp(-slack_centered / 0.04753091717408599))
    slack_penalty = 4.411180886877736 * np.maximum(0.0, -slack_centered) - 0.9789412310097287 * np.maximum(0.0, slack_centered)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.4567059748693519 * normalize(inv_energy) * ddl_gate
    rank_score = -1.228978126614614 * normalize(rank + eps) * ddl_gate
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 1.7771781625142415 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_sat = 1.0 - np.exp(-0.30276545667107574 * wait)
    wait_score = -normalize(wait_sat + eps) * (1.0 - ddl_gate)
    unc_slack_interaction = 3.8191350662480756 * uncert_norm * np.maximum(0.0, -slack_centered) * (1.0 - ddl_gate)
    successor_release_score = -0.1743298764626274 * normalize(work + eps) * rank * ddl_gate
    system_load_proxy = normalize(duration + eps) + normalize(uncert + eps)
    load_gate = np.clip(system_load_proxy - 0.756260374141318, 0.0, 1.0)
    host_load_penalty = 0.756260374141318 * load_gate * energy_score
    score = slack_penalty + energy_score + rank_score + dur_score + wait_score + unc_slack_interaction + successor_release_score + host_load_penalty
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 62310.32225317356
    min_safe = -finfo.max / 62310.32225317356
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
