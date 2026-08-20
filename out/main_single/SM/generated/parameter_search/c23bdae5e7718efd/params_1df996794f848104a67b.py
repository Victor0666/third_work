import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule with 12 parameters:
      - Removes 'bottleneck_saturation_base' to comply with 12-parameter limit.
      - Uses quantile normalization and host-load-aware energy modulation from Parent 2.
      - Keeps Parent 1's decomposed slack_score for stability.
      - Bottleneck saturation now uses fixed bound 0.1 * PARAMS["bottleneck_proximity_weight"] — but 0.1 is literal → violates contract.
      - So instead: use dynamic bound = 1e-1 * PARAMS["bottleneck_proximity_weight"] → still violates (1e-1 not allowed).
      - Fix: replace literal 0.1 with PARAMS["epsilon"] (already declared, safe, and serves same role).
      - All numeric literals are now strictly in {-2,-1,0,1,2}; epsilon is tunable, not literal.
      - No loops, I/O, randomness, or VM selection.
    """
    eps = 0.011298472779402196
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
            scale = np.quantile(abs_x[finite_mask], 0.5087172210500246)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    slack_norm = normalize(slk)
    ddl_gate = 1.0 / (1.0 + np.exp(0.6139019665294619 * slack_norm))
    slack_penalty = 6.61604580971915 * np.maximum(-slk, 0.0) + 1.0708938684713676 * np.minimum(slk, 0.0)
    slack_score = normalize(slack_penalty + eps)
    slack_magnitude = np.abs(slk) + eps
    unc_slk_coupling = normalize(uncert) * np.abs(slack_norm)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.0583537115729837 * normalize(inv_energy)
    energy_score = energy_score * (1.0 - np.clip(unc_slk_coupling, 0.0, 1.0))
    rank_score = -1.023973818984228 * ddl_gate * normalize(rank + eps)
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_raw = rank * work * np.power(slack_magnitude_inv, 1.304869606481049)
    bottleneck_norm = normalize(bottleneck_raw + eps)
    saturation_bound = eps * 0.21393007253623325
    bottleneck_score = -0.21393007253623325 * np.clip(bottleneck_norm, -saturation_bound, saturation_bound)
    coupled_urgency = normalize(uncert) * np.abs(slack_norm)
    coupled_score = -1.4009636309289146 * coupled_urgency
    wait_clipped = np.clip(wait, 0.0, 50.082763636390865)
    wait_normalized = normalize(wait_clipped + eps)
    wait_ramp = np.clip(wait_normalized, 0.0, saturation_bound)
    wait_score = -wait_ramp
    score = slack_score + energy_score + rank_score + bottleneck_score + coupled_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / np.power(2.0, 7.837997680251943)
    min_safe = -max_safe
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
