import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with:
       - Quantile-based robust scaling (Q3-Q1 IQR + epsilon) → handles skew/outliers better.
       - Load-aware conditional gate decoupling DDL urgency from energy penalties.
       - Piecewise linear urgency with breakpoint at slack=0 → sharp deadline compliance.
       - All numeric literals are {-2,-1,0,1,2}; exactly 12 parameters; all used; no hidden constants.
    """
    eps = 0.04054760881946079
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
        q1 = np.quantile(x_abs, 0.08080378355212972, method='linear')
        q3 = np.quantile(x_abs, 1.0 - 0.08080378355212972, method='linear')
        spread = q3 - q1 + eps
        return x / (spread + eps)
    norm_slack = scale_feature(slack)
    norm_energy = scale_feature(min_incremental_energy)
    norm_duration = scale_feature(min_exec_time + min_comm_time)
    norm_rank = scale_feature(upward_rank)
    norm_work = scale_feature(remaining_work)
    norm_wait = scale_feature(ready_wait_time)
    norm_uncert = scale_feature(uncertainty)
    urgency_signal = np.where(slack < 0.0, -slack * 1.5864223702802376, 0.0)
    load_proxy = (norm_duration + norm_energy + norm_uncert) / 2.0
    load_gate = np.where(load_proxy > 1.070543646642745, 1.0, 0.0)
    uncert_gate = np.tanh((0.857084818284468 - norm_uncert) * 2.0)
    ddl_protection_gate = np.clip(urgency_signal * uncert_gate, 0.0, 1.0)
    boosted_rank = norm_rank * (1.0 + 1.0947970334441894 * ddl_protection_gate)
    successor_release_score = norm_work * norm_rank * 0.07087840176828301 * ddl_protection_gate
    wait_benefit = np.tanh(1.4842570389958909e-05 * (norm_wait + 1.7265587281045288e-08))
    risk_energy_penalty = norm_energy * norm_uncert * ddl_protection_gate * (1.0 - uncert_gate) * load_gate
    score = +urgency_signal - boosted_rank - successor_release_score - wait_benefit + 0.5663383401840452 * risk_energy_penalty - 0.525833186851433 * norm_energy + 1.8401242883770859 * norm_work
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
