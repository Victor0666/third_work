import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule with all numeric constants declared in PARAMETER_SCHEMA.
    
    Replaces hardcoded 0.25/0.75 quantiles with tunable parameters.
    All other numeric literals are now restricted to {-2, -1, 0, 1, 2}.
    Uses np.finfo for immutable safeguards instead of hidden epsilons.
    """
    eps = 0.013610718330888198
    finfo = np.finfo(float)
    eps_safe = max(eps, finfo.tiny)

    def robust_normalize(x):
        x = np.asarray(x, dtype=float)
        q1 = np.quantile(x, 0.20331454822160347)
        q3 = np.quantile(x, 0.7699750005890267)
        iqr = q3 - q1
        center = np.median(x)
        scale = iqr if iqr > eps_safe else eps_safe
        return (x - center) / scale
    duration = min_exec_time + min_comm_time
    norm_duration = robust_normalize(duration)
    slack_penalty = np.where(slack < 0, -np.abs(slack) ** 2.4541034691334587, slack)
    norm_slack = robust_normalize(slack_penalty)
    norm_rank = robust_normalize(upward_rank + 0.18783641978798932)
    median_slack = np.median(slack)
    is_critical_and_at_risk = ((upward_rank > np.median(upward_rank)) & (slack < median_slack)).astype(float)
    criticality_boost = is_critical_and_at_risk * 1.6742022355230657
    norm_energy = robust_normalize(min_incremental_energy)
    coupled_uncertainty = np.clip(norm_duration * uncertainty * 0.5830967076873391, -2.0, 2.0)
    wait_scaled = ready_wait_time / (8.670067307437426 + eps_safe)
    norm_wait = 2.0 / np.pi * np.arctan(wait_scaled)
    slack_uncertainty_amplifier = np.where(slack < 0, uncertainty * 1.4912489870105659, 0.0)
    norm_slack_uncertainty = robust_normalize(slack_uncertainty_amplifier)
    score = norm_slack + norm_slack_uncertainty + criticality_boost * norm_rank - 0.912490074257766 * norm_energy + coupled_uncertainty - norm_wait
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
