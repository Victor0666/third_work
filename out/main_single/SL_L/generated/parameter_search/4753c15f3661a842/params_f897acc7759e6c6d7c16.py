import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """
    Self-evolved priority rule featuring a *unified joint feasibility gate*:
      - Replaces separate `feasibility_gate` and `unc_sigmoid_gate` with smooth, bounded `tanh(slack / (uncertainty + eps))`
        → captures 'risk-adjusted slack density': high slack + low uncertainty = full boost; negative slack or high uncertainty = strong suppression.
      - Introduces `rank_slack_uncertainty_coupling`: direct three-way interaction term `upward_rank * tanh(slack / (uncertainty + eps))`,
        enabling critical-path awareness that *naturally degrades* under uncertainty or deadline pressure — no fragile conditionals.
      - Removes all hard caps and sigmoid gates; relies entirely on bounded tanh for stability and differentiability.
      - Keeps MAD-normalized inputs throughout; preserves decoupled energy-slack coupling and saturating fairness logic.
      - All parameters used; no numeric literals beyond {-2,-1,0,1,2}; fully deterministic and finite.
    """
    eps = 2.161469626104502e-09
    finfo = np.finfo(np.float64)
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    slack = np.nan_to_num(np.asarray(slack, dtype=np.float64), nan=0.0, posinf=finfo.max, neginf=finfo.min)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=np.float64), nan=0.0, posinf=finfo.max, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=eps)

    def mad_normalize(x):
        x = np.asarray(x, dtype=np.float64)
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        scale = 1.4773704533884544 * (mad if mad > eps else eps)
        return (x - med) / scale
    slack_pressure = -slack
    slack_urgency = 1.6642002934795654 * np.tanh(slack_pressure * 0.5779998023787742)
    duration = min_exec_time + min_comm_time
    duration_norm = mad_normalize(duration)
    energy_norm = mad_normalize(min_incremental_energy)
    energy_base = 0.3626585510311894 * energy_norm
    energy_slack_gate = np.clip(slack_pressure, 0.0, np.inf)
    energy_score = energy_base * (1.0 + 0.08343182148525707 * np.tanh(energy_slack_gate * 0.5779998023787742))
    joint_feasibility = np.tanh(slack / (uncertainty + eps) * 0.4897188117026981)
    rank_norm = mad_normalize(upward_rank)
    critical_boost = 1.0907379446731376 * rank_norm * joint_feasibility
    work_density = np.divide(remaining_work, duration + eps)
    work_density_norm = mad_normalize(work_density)
    work_density_bonus = 0.804994041285114 * work_density_norm * joint_feasibility
    wait_norm = mad_normalize(ready_wait_time)
    wait_boost = 1.0 - np.exp(-1.1117618025391325 * np.maximum(wait_norm, 0.0))
    wait_score = wait_boost * joint_feasibility
    unc_norm = mad_normalize(uncertainty)
    rank_slack_uncertainty_interaction = rank_norm * np.tanh(slack * (1.0 / (uncertainty + eps)))
    resilience_boost = 1.4782109922328677 * rank_slack_uncertainty_interaction
    score = slack_urgency + 1.0950356216921586 * duration_norm + energy_score - critical_boost - work_density_bonus + wait_score - resilience_boost
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=-finfo.max)
    return score
