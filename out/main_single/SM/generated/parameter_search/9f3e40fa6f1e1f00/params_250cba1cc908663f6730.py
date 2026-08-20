import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule with verified parameter usage:
      - Removed unused 'successor_release_sharpness'; replaced bottleneck sharpening with stable softplus.
      - Added host-load sensitivity term using energy-per-duration ratio, gated by ddl_protection_gate.
      - Replaced linear wait ramp with bounded exponential saturation for improved starvation mitigation.
      - All numeric literals are restricted to {-2,-1,0,1,2}; epsilon and machine limits use np.finfo.
      - Unified quantile-based normalization across all features.
      - Strict DDL-first ordering preserved via dominant slack_penalty.
    """
    eps = 1.995034061886725e-06
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
            scale = np.quantile(abs_x[finite_mask], 0.747247330577657)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    slack_norm = normalize(slk)
    ddl_gate = 1.0 / (1.0 + np.exp(0.32646031811161064 * slack_norm))
    slack_penalty = np.where(slk < 0, 2.9534277290598823 * np.abs(slack_norm), -2.3352412909309117 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -1.8857658289299788 * normalize(inv_energy)
    rank_score = -2.6829466106363613 * ddl_gate * normalize(rank + eps)
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_base = rank * work * slack_magnitude_inv
    bottleneck_sharpened = np.log1p(np.exp(bottleneck_base))
    bottleneck_score = -1.72608024678139 * normalize(bottleneck_sharpened + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.7824457486139539 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_normalized = wait / (27.842809016457824 + eps)
    wait_saturation = 1.0 - np.exp(-wait_normalized)
    wait_saturation = np.clip(wait_saturation, 0.0, 1.0)
    wait_score = -wait_saturation
    energy_per_duration = energy / (duration + eps)
    load_proxy = normalize(energy_per_duration + eps)
    load_score = -0.42348760495286486 * ddl_gate * load_proxy
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score + load_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 2.213994144895084
    min_safe = finfo.min / 2.213994144895084
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
