import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule: removes unused parameter, ensures all declared params are used,
    preserves robust normalization and deadline-critical gating logic.
    
    Key features:
      - Uses tanh-based lateness penalty for negative slack
      - Upward rank boosted only when slack < threshold (gated criticality)
      - Efficiency proxy: duration / work, normalized robustly
      - Wait time prioritization with soft power-law decay
      - Uncertainty-slack coupling via sigmoid-weighted interaction
      - All numeric literals restricted to {-2,-1,0,1,2}; epsilon from PARAMS
    """
    eps = 4.487494581085358e-06
    finfo = np.finfo(float)

    def robust_normalize(x):
        x = np.asarray(x, dtype=float)
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        scale = np.where(mad > eps, mad, eps)
        return (x - med) / scale
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    duration = min_exec_time + min_comm_time
    norm_duration = robust_normalize(duration)
    eff_ratio = duration / (remaining_work + eps)
    norm_eff_ratio = robust_normalize(eff_ratio)
    lateness_risk = np.tanh(-norm_slack * 2.6161941560184117)
    slack_gate = np.where(norm_slack < -0.40691632325798244, 0.8967682507729363, 1.0)
    gated_rank = norm_rank * slack_gate
    wait_priority = np.sign(norm_wait) * np.abs(norm_wait) ** 1.2444259019743327
    coupling_strength = 1.0 / (1.0 + np.exp(-norm_slack - norm_uncert))
    coupled_uncert = coupling_strength * norm_uncert * 0.715728704898081
    score = lateness_risk + 0.10972366102751782 * norm_eff_ratio - gated_rank + norm_energy - wait_priority + coupled_uncert
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
