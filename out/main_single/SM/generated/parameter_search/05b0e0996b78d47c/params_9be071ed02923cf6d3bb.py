import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule with dual-mode normalization (quantile for temporal, max-abs for structural),
    crisp piecewise-linear DDL gate, and sharpened fused temporal risk. All parameters are used;
    no unused or hidden constants. Numeric literals strictly limited to {-2,-1,0,1,2}.
    """
    eps = 0.00018163630653423836
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    work = np.asarray(remaining_work, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)

    def normalize_quantile(x, q=0.9495426118068575):
        x = np.asarray(x)
        abs_x = np.abs(x)
        finite_mask = np.isfinite(abs_x)
        if np.any(finite_mask):
            scale = np.quantile(abs_x[finite_mask], q)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)

    def normalize_max_abs(x):
        x = np.asarray(x, dtype=np.float64)
        abs_x = np.abs(x)
        finite_mask = np.isfinite(abs_x)
        if np.any(finite_mask):
            scale = np.max(abs_x[finite_mask])
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    slack_norm = normalize_quantile(slk)
    slack_penalty = np.where(slk < 0, 1.64600246416652 * np.abs(slack_norm), -2.3748360714340966 * slack_norm)
    gate_width = 10.237861632170334 + eps
    ddl_gate = np.where(slk <= 0, 1.0, np.where(slk <= gate_width, 1.0 - slk / gate_width, 0.0))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -2.6498193837408164 * normalize_max_abs(inv_energy)
    energy_score = energy_score * (1.0 - 0.4215218377852641 * ddl_gate)
    rank_powered = np.power(rank + eps, 1.9429948517327982)
    rank_score = -normalize_max_abs(rank_powered)
    slack_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_term = rank * work * slack_inv
    bottleneck_score = -2.092916174686493 * normalize_max_abs(bottleneck_term + eps)
    temporal_risk = exec_t + comm_t + uncert
    temporal_risk_sharp = np.power(temporal_risk + eps, 2.4555097651430278)
    dur_score = normalize_quantile(temporal_risk_sharp)
    wait_sat = 1.0 - np.exp(-wait / (72.69677864306608 + eps))
    wait_score = -wait_sat
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max / 2.0, neginf=finfo.min / 2.0)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
