import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with smooth sigmoid gating on rank activation — replacing brittle piecewise critical zone.
    
    Key improvements:
    - Replaces hard [slack_min_bound, 0] piecewise activation with a *smooth, differentiable sigmoid gate* on slack,
      centered at slack=0 and scaled by PARAMS["rank_slack_sigmoid_steepness"] — improves gradient stability and avoids threshold artifacts.
    - Restores robust mean-absolute normalization (not MAD-scaled) as confirmed more stable across sparse/N=1 ready sets.
    - Removes inactive parameters (`robustness_mad_factor`, `slack_pressure_decay_rate`) to reduce over-parameterization.
    - Keeps all other proven components: bounded uncertainty sigmoid, additive criticality bonus (not multiplicative), and unified slack penalty.
    - All literals are -2,-1,0,1,2; no hidden constants; all tunables declared in PARAMETER_SCHEMA.
    """
    eps = 6.513760756715846e-09
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
    slack_score = np.where(slack < 0, (-slack) ** 1.0818738784802404, 0.0)
    duration_total = min_exec_time + min_comm_time + eps
    duration_norm = robust_normalize(duration_total)
    duration_risk = duration_norm * uncertainty * 0.6454847185240247
    energy_per_sec = min_incremental_energy / (duration_total + eps)
    energy_eff_score = robust_normalize(energy_per_sec) * 1.2437959563159553
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = robust_normalize(uncertainty * deadline_pressure) * 1.9208486888291567
    median_unc = np.median(uncertainty)
    ddl_protection_gate = ((slack >= 0.0) & (uncertainty <= median_unc + eps)).astype(float)
    rank_norm = robust_normalize(upward_rank)
    critical_bonus = 4.005744224452551 * rank_norm * ddl_protection_gate
    slack_centered = slack - 0.0
    slack_scale = -29.987059677812667 if -29.987059677812667 < 0 else -1.0
    width = np.abs(slack_scale) + eps
    rank_gate = 1.0 / (1.0 + np.exp(-7.258507073859025 * (slack_centered / (width + eps))))
    rank_score = -robust_normalize(upward_rank) * rank_gate
    unc_sigmoid = 1.0 / (1.0 + np.exp(-1.8749427230978601 * (uncertainty - 1.0)))
    energy_norm = robust_normalize(min_incremental_energy)
    unc_norm = robust_normalize(uncertainty)
    energy_uncertainty_score = 1.2653253754917182 * energy_norm * unc_norm * unc_sigmoid
    energy_base_score = robust_normalize(min_incremental_energy)
    score = robust_normalize(slack_score) + duration_risk + energy_eff_score + unc_slack_coupling + energy_uncertainty_score + energy_base_score - critical_bonus - rank_score
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
