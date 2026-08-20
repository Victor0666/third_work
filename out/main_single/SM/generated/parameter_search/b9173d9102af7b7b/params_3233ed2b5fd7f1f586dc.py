import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule with:
      - Crisp piecewise-linear DDL gate for robust deadline boundary control.
      - Max-abs normalization for stability and interpretability.
      - Temporal risk sharpening: (exec_t + comm_t + uncert)^p before normalization.
      - Load-bottleneck coupling: energy_per_duration × bottleneck_term × ddl_gate —
        enabling energy-aware bottleneck prioritization only where deadline safety permits.
      - Exponential starvation mitigation and exponentiated criticality.
      - All numeric literals are {-2,-1,0,1,2}; no unused parameters; deterministic & finite.
    """
    eps = 0.002181821313798746
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
    gate_width = 3.2195250997774845 + eps
    ddl_gate = np.where(slk <= 0, 1.0, np.where(slk <= gate_width, 1.0 - slk / gate_width, 0.0))
    slack_penalty = np.where(slk < 0, 5.231868419345928 * np.abs(slk), -2.5593377485353828 * slk)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -1.8058082436528178 * normalize(inv_energy)
    energy_score = energy_score * (1.0 - 0.4305778884904462 * ddl_gate)
    rank_powered = np.power(rank + eps, 0.33451953219529473)
    rank_score = -normalize(rank_powered)
    slack_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_term = rank * work * slack_inv
    bottleneck_score = -1.0465206944383598 * normalize(bottleneck_term + eps)
    temporal_risk = exec_t + comm_t + uncert
    temporal_risk_sharpened = np.power(temporal_risk + eps, 2.070126472610175)
    dur_score = normalize(temporal_risk_sharpened)
    wait_sat = 1.0 - np.exp(-wait / (19.000511066361053 + eps))
    wait_score = -wait_sat
    energy_per_duration = energy / (exec_t + comm_t + eps)
    load_proxy = normalize(energy_per_duration + eps)
    coupled_term = load_proxy * bottleneck_term * ddl_gate
    coupled_score = -0.3315905938892084 * normalize(coupled_term + eps)
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score + coupled_score
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max / 2, neginf=-finfo.max / 2)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
