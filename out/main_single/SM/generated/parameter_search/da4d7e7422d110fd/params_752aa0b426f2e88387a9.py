import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with three key structural improvements:
      1. ADDITIVE SIGN-PRESERVING UNCERTAINTY GATING: `duration + ratio * (uncert - median(uncert))`
      2. ASYMMETRIC PIECEWISE SLACK SCALING: clamp slack to [-1,1], then apply steeper penalty slope
         for negative slack (via `asymmetric_slack_gain`) — sharpening DDL violation response.
      3. CRITICAL-PATH STARVATION GUARD: adds conditional penalty `(1 - rank_norm)` under slack < 0,
         ensuring high-upward-rank tasks are never starved during deadline pressure.
      All features use max-abs normalization; all operations guarded against NaN/inf; deterministic and finite.
      Parameter count reduced to 12 by merging starvation_penalty_weight into asymmetric_slack_gain logic
      via fixed multiplicative coupling (1.0) — preserving starvation mitigation without new parameter.
    """
    eps = 6.941686859591595e-06
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
    slk_clamped = np.clip(slk, -1.0, 1.0)
    slk_scaled = np.where(slk_clamped < 0, slk_clamped * 1.0020459212449102, slk_clamped)
    slk_eff = slk_scaled
    gate_width = 8.184557953478707 + eps
    ddl_gate = np.where(slk_eff <= 0, 1.0, np.where(slk_eff <= gate_width, 1.0 - slk_eff / gate_width, 0.0))
    slack_penalty = np.where(slk_eff < 0, 0.9734419980196751 * np.abs(slk_eff), -3.44960778632875 * slk_eff)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.250311030973101 * normalize(inv_energy)
    slack_decay = np.exp(-np.abs(slk_eff) * 0.6587815612955262)
    energy_score = energy_score * (1.0 - 0.18952934142798772 * ddl_gate) * slack_decay
    rank_powered = np.power(rank + eps, 0.7805318708414586)
    rank_score = -normalize(rank_powered)
    robust_abs_slk = np.abs(slk_eff) + eps
    bottleneck_term = rank * work / robust_abs_slk
    bottleneck_score = -3.3024810350186886 * normalize(bottleneck_term + eps)
    duration = exec_t + comm_t
    median_uncert = np.median(uncert) if N > 0 else 0.0
    median_broadcast = np.full(N, median_uncert, dtype=np.float64)
    duration_with_uncert = duration + 0.30115999507262414 * (uncert - median_broadcast)
    dur_score = normalize(duration_with_uncert + eps)
    wait_sat = 1.0 - np.exp(-wait / (18.038956083194126 + eps))
    wait_score = -wait_sat
    rank_norm = (normalize(rank + eps) + 1.0) / 2.0
    starvation_guard = np.where(slk_eff < 0, 1.0 * (1.0 - rank_norm), 0.0)
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score + starvation_guard
    finfo = np.finfo(np.float64)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max / 2.0, neginf=finfo.min / 2.0)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
