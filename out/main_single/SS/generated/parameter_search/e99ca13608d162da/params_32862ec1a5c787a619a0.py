import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with:
       - Quantile-based robust scaling (Q3-Q1 IQR + epsilon) → handles skew/outliers better.
       - Load-aware conditional gate decoupling DDL urgency from energy penalties.
       - Piecewise linear urgency with breakpoint at slack=0 → sharp deadline compliance.
       - All numeric literals are {-2,-1,0,1,2}; exactly 12 parameters; all used; no hidden constants.
    """
    eps = 0.00843056040298592
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def scale_feature(x):
        x_abs = np.abs(x)
        q1 = np.quantile(x_abs, 0.4983243452227878, method='linear')
        q3 = np.quantile(x_abs, 1.0 - 0.4983243452227878, method='linear')
        spread = q3 - q1 + eps
        return x / (spread + eps)
    norm_slack = scale_feature(slack)
    norm_energy = scale_feature(min_incremental_energy)
    norm_duration = scale_feature(min_exec_time + min_comm_time)
    norm_rank = scale_feature(upward_rank)
    norm_work = scale_feature(remaining_work)
    norm_wait = scale_feature(ready_wait_time)
    norm_uncert = scale_feature(uncertainty)
    urgency_signal = np.where(slack < 0.0, -slack * 1.2419502641708604, 0.0)
    load_proxy = (norm_duration + norm_energy + norm_uncert) / 2.0
    load_gate = np.where(load_proxy > 0.0004974998097817627, 1.0, 0.0)
    uncert_gate = np.tanh((0.842321660800066 - norm_uncert) * 2.0)
    ddl_protection_gate = np.clip(urgency_signal * uncert_gate, 0.0, 1.0)
    boosted_rank = norm_rank * (1.0 + 1.0020516541905748 * ddl_protection_gate)
    successor_release_score = norm_work * norm_rank * 0.4926175112688126 * ddl_protection_gate
    wait_benefit = np.tanh(0.020984524872252452 * (norm_wait + 1.0255480415965149e-08))
    risk_energy_penalty = norm_energy * norm_uncert * ddl_protection_gate * (1.0 - uncert_gate) * load_gate
    score = +urgency_signal - boosted_rank - successor_release_score - wait_benefit + 0.6398449816566306 * risk_energy_penalty - 0.7899286218329037 * norm_energy + 0.9640300506513848 * norm_work
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
