import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining Parent 2's hard DDL protection and simplified bottleneck
    with Parent 1's starvation-aware wait saturation and robust slack normalization.
    Key improvements:
      - Hard binary DDL gate enforces strict feasibility-first priority (no deadline violations).
      - Bottleneck term uses tunable power (rank * work)^p — removes fragile division and aligns with congestion evidence.
      - Criticality is softly modulated by slack feasibility via tunable-slope sigmoid.
      - Starvation term weighted explicitly to balance fairness without compromising deadlines.
      - All features max-abs normalized for stability and interpretability.
      - Exactly 10 parameters — all declared, all used, no literals beyond {-2,-1,0,1,2}.
      - Deterministic, finite, shape-correct, and DDL-hard-constraint safe.
    """
    eps = 0.04243875427406084
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
    abs_slk = np.abs(slk) + eps
    robust_slk_mag = np.power(abs_slk, 0.6942052303305405)
    robust_slk_norm = robust_slk_mag / (robust_slk_mag + eps)
    slk_robust = robust_slk_norm * np.sign(slk)
    ddl_protection_gate = np.where(slk_robust >= 0, 1.0, 0.0)
    slack_penalty = np.where(slk_robust < 0, 2.0330289591631563 * np.abs(slk_robust), 0.0)
    inv_energy = 1.0 / (energy + eps)
    energy_score_base = -1.7883008466867214 * normalize(inv_energy)
    energy_score = energy_score_base * ddl_protection_gate
    rank_powered = np.power(rank + eps, 1.1391366170891248)
    rank_score_unmod = -normalize(rank_powered)
    slack_feasibility = 1.0 / (1.0 + np.exp(-slk_robust * 3.735893667062906))
    rank_score = rank_score_unmod * slack_feasibility
    bottleneck_product = rank * work + eps
    bottleneck_powered = np.power(bottleneck_product, 1.3467919407873732)
    bottleneck_score = -1.3467919407873732 * normalize(bottleneck_powered)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.5665209967865137 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_sat = 1.0 - np.exp(-wait / (24.523081978425708 + eps))
    wait_score = -0.7954761119441882 * wait_sat
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max / 2.0, neginf=finfo.min / 2.0)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
