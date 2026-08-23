import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's robust sigmoid gating and Parent 1's fairness & density terms.
    
    Key structural improvements:
    - Integrates wait_fairness_gain and work_density_weight from Parent 1 to prevent starvation and reward throughput-critical paths.
    - Retains Parent 2's smooth sigmoid rank gating and mean-absolute normalization for stability across N=1 and sparse sets.
    - Introduces *normalized wait-time saturation* using exp(-gain * norm_wait) instead of linear wait_boost, avoiding unbounded growth.
    - Replaces raw work_density with robustly normalized (remaining_work / (exec+comm+eps)) to ensure scale-invariance.
    - All beneficial terms subtracted; all penalties added; strict DDL-first ordering preserved via slack_score dominance.
    """
    eps = 8.506224889187908e-09
    finfo = np.finfo(float)
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=finfo.max, neginf=finfo.min)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=finfo.max, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=eps, posinf=finfo.max, neginf=eps)
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=float)

    def robust_normalize(x):
        x = np.asarray(x, dtype=float)
        abs_x = np.abs(x)
        scale = np.mean(abs_x) + eps
        return x / (scale + eps)
    slack_score = np.where(slack < 0, (-slack) ** 2.1012835946158863, 0.0)
    duration_total = min_exec_time + min_comm_time + eps
    duration_norm = robust_normalize(duration_total)
    duration_risk = duration_norm * uncertainty * 0.23628933672511004
    energy_per_sec = min_incremental_energy / (duration_total + eps)
    energy_eff_score = robust_normalize(energy_per_sec) * 0.45360041733352974
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = robust_normalize(uncertainty * deadline_pressure) * 1.8065523362351419
    median_unc = np.median(uncertainty)
    ddl_protection_gate = ((slack >= 0.0) & (uncertainty <= median_unc + eps)).astype(float)
    rank_norm = robust_normalize(upward_rank)
    critical_bonus = 2.0282323227411956 * rank_norm * ddl_protection_gate
    slack_centered = slack - 0.0
    width = np.abs(2.1012835946158863) + eps
    rank_gate = 1.0 / (1.0 + np.exp(-3.5092852653838986 * (slack_centered / (width + eps))))
    rank_score = -robust_normalize(upward_rank) * rank_gate
    unc_sigmoid = 1.0 / (1.0 + np.exp(-2.1819679004764203 * (uncertainty - 1.0)))
    energy_norm = robust_normalize(min_incremental_energy)
    unc_norm = robust_normalize(uncertainty)
    energy_uncertainty_score = 0.0013849099815235312 * energy_norm * unc_norm * unc_sigmoid
    energy_base_score = robust_normalize(min_incremental_energy)
    wait_norm = robust_normalize(ready_wait_time)
    wait_boost = 1.0 - np.exp(-0.0827180158070733 * np.maximum(0.0, wait_norm))
    work_density = remaining_work / (duration_total + eps)
    work_density_norm = robust_normalize(work_density)
    work_density_bonus = 0.8029909407632233 * work_density_norm
    score = robust_normalize(slack_score) + duration_risk + energy_eff_score + unc_slack_coupling + energy_uncertainty_score + energy_base_score + wait_boost - critical_bonus - rank_score - work_density_bonus
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
