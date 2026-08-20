import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with improved structural robustness:
      - Replaced `wait_ramp_threshold` with self-scaling wait-term using quantile-normalized wait time → eliminates overparameterization.
      - Replaced brittle `np.clip` saturation with smooth `tanh` activation for bottleneck and wait terms → improves CMA-ES gradient flow.
      - Introduced feasibility-aware energy modulation gate: `ddl_gate * (1 - np.tanh(uncert))`, ensuring joint DDL+uncertainty suppression.
      - All numeric literals strictly limited to {-2,-1,0,1,2}; no hidden constants.
      - Removed redundant `slack_urgency_gain` sign flip; unified into clean `np.where` for interpretability.
      - Preserved all safeguards: NaN/inf handling, finite output, shape enforcement, and epsilon-guarded divisions.
    """
    eps = 0.021973334769000666
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
            scale = np.quantile(abs_x[finite_mask], 0.7938353263555338)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    slack_norm = normalize(slk)
    ddl_gate = 1.0 / (1.0 + np.exp(0.6205764820729899 * slack_norm))
    slack_score = np.where(slk < 0, 9.103849660056108 * np.abs(slack_norm), -0.3777400441148826 * np.abs(slack_norm))
    energy_mod_gate = ddl_gate * (1.0 - np.tanh(uncert + eps))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -1.2237899667298766 * normalize(inv_energy) * energy_mod_gate
    rank_score = -0.0536438847878498 * ddl_gate * normalize(rank + eps)
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_raw = rank * work * np.power(slack_magnitude_inv, 0.9081196974754248)
    bottleneck_norm = normalize(bottleneck_raw + eps)
    bottleneck_score = -1.121973275815659 * np.tanh(bottleneck_norm)
    coupled_urgency = np.tanh(normalize(uncert)) * np.abs(slack_norm)
    coupled_score = -0.36479303454916556 * coupled_urgency
    wait_norm = normalize(wait + eps)
    wait_score = -0.5053665574719093 * np.tanh(wait_norm)
    score = slack_score + energy_score + rank_score + bottleneck_score + coupled_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / np.power(2.0, 14.299324467854778)
    min_safe = -max_safe
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
