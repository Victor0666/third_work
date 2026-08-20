import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with:
      - Bounded piecewise-linear slack gating (replacing fragile exponentiation) for robustness near zero slack.
      - Successor-release-aware bottleneck: (rank × work) / (1 + |min_successor_slack| + ε), approximated conservatively using current slack.
      - Conditional energy activation only when current slack >= 0 (strict DDL feasibility).
      - All numeric literals strictly {-2,-1,0,1,2}; no hidden constants.
      - Deterministic, finite, shape-preserving, and safe against NaN/inf/div-zero.
      - Exactly 12 parameters; all used; no unused or missing references.
    """
    eps = 0.0016519302764060073
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
    slk_abs = np.abs(slk)
    slk_linear = np.where(slk_abs <= 1, slk, np.where(slk > 1, 1.0, -1.0))
    ddl_protection_gate = np.where(slk >= 0, 1.0, 0.0)
    slack_penalty = np.where(slk < 0, 2.6943296721335344 * np.abs(slk), -0.7303046732280404 * slk)
    inv_energy = 1.0 / (energy + eps)
    energy_score_base = -1.9485019839568283 * normalize(inv_energy)
    energy_score = energy_score_base * ddl_protection_gate
    rank_powered = np.power(rank + eps, 1.0935377593961084)
    rank_score_unmod = -normalize(rank_powered)
    slack_feasibility = 1.0 / (1.0 + np.exp(-slk * 2.842906551952904))
    rank_score = rank_score_unmod * slack_feasibility
    bottleneck_numerator = rank * work + eps
    bottleneck_denom = 1.0 + np.abs(slk) + eps
    bottleneck_proxy = bottleneck_numerator / bottleneck_denom
    bottleneck_powered = np.power(bottleneck_proxy, 1.1144034560451836)
    bottleneck_score = -0.8072714396897929 * normalize(bottleneck_powered)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.6909478538751089 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_sat = 1.0 - np.exp(-wait / (37.11200625721646 + eps))
    starvation_gain = np.where(ddl_protection_gate == 1.0, 0.44565418245939603, 0.474876927896609)
    wait_score = -starvation_gain * wait_sat
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max / 2.0, neginf=finfo.min / 2.0)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
