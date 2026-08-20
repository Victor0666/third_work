import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining robust deadline risk containment (Parent 2),
    median-stabilized normalization (Performance Analysis), slack-gated criticality,
    and successor-release risk coupling.
    
    Key improvements:
      - Uses median absolute deviation (MAD) normalization for all features → stable under skew/outliers
      - Criticality gating uses slack margin relative to median positive slack → adaptive deadline awareness
      - Successor-release interaction: tanh-saturated product of normalized successor work and inverse slack
      - Preserves inverse-energy efficiency and exponential wait saturation from Parent 2
      - All operations guarded against NaN/inf/zero; deterministic and finite output.
    """
    eps = 2.7376328075191696e-09
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    work = np.asarray(remaining_work, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)

    def mad_normalize(x):
        x = np.asarray(x)
        med = np.median(x)
        mad = np.median(np.abs(x - med)) + eps
        return (x - med) / mad
    duration = exec_t + comm_t
    norm_duration = mad_normalize(duration)
    inv_energy = 1.0 / (energy + eps)
    norm_inv_energy = mad_normalize(inv_energy)
    energy_score = -2.1939821855204573 * norm_inv_energy
    norm_slack = mad_normalize(slk)
    slack_penalty = np.where(slk < 0, 2.122141545586756 * norm_slack ** 2, -1.5199729127592632 * np.abs(norm_slack))
    pos_slack = slk[slk > eps]
    median_pos_slack = np.median(pos_slack) if len(pos_slack) > 0 else 1.0
    slack_margin = 0.41054254634296544 * max(median_pos_slack, eps)
    rank_gate = np.where(slk > -slack_margin, 1.0, 0.0)
    norm_rank = mad_normalize(rank + eps)
    rank_score = -0.0320399847990219 * norm_rank * rank_gate
    norm_uncert = mad_normalize(uncert + eps)
    dur_uncert_blend = (1.0 + 1.4129225356335393 * norm_uncert) / (np.abs(norm_duration) + eps + 1.4129225356335393 * np.abs(norm_uncert) + eps)
    dur_score = mad_normalize(dur_uncert_blend)
    wait_sat = 1.0 - np.exp(-0.0075476630506781965 * wait)
    norm_wait_sat = mad_normalize(wait_sat + eps)
    wait_score = -norm_wait_sat
    slack_stress = np.where(slk < 0, np.abs(norm_slack), 0.0)
    unc_slack_interaction = 0.16256308405516146 * norm_uncert * slack_stress
    work_slack_product = work * np.where(slk > eps, 1.0 / (slk + eps), 0.0)
    norm_work_slack = mad_normalize(work_slack_product + eps)
    successor_risk = 1.233503737936375 * np.tanh(norm_work_slack)
    score = slack_penalty + energy_score + rank_score + dur_score + wait_score + unc_slack_interaction + successor_risk
    finfo = np.finfo(np.float64)
    score = np.nan_to_num(score, nan=np.median(score) if np.any(np.isfinite(score)) else 0.0, posinf=finfo.max / 22578698.320289817, neginf=finfo.min / 1309947.4261714285)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
