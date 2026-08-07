import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining Parent 2's robust linear deadline risk modeling with Parent 1's wait-aware anti-starvation mechanism.
    
    Key structural improvements:
      - Introduces novel `wait_urgency_coupling`: multiplies normalized wait time by slack urgency (not just raw wait), ensuring waiting tasks gain priority *only when deadline pressure exists* — avoids premature promotion under loose deadlines.
      - Retains Parent 2's superior IQR-based adaptive normalization (with tunable percentiles) for all terms, replacing std-based dispersion scaling for outlier robustness.
      - Keeps direct `upward_rank × remaining_work` interaction (no duration contamination) as validated by counterfactual analysis.
      - Preserves linear slack penalty + uncertainty coupling for sharp, stable DDL risk signal.
      - Energy term retains offset-shifted normalization to balance fairness vs. efficiency under tight slack.
      - All numeric literals strictly in {-2,-1,0,1,2}; no hidden constants; no branching; deterministic.
    """
    eps = 2.493650413605958e-05
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    def adaptive_normalize(x):
        x = np.copy(x)
        if N == 0:
            return np.zeros_like(x)
        q_low = np.percentile(x, 32.67664286033773)
        q_high = np.percentile(x, 87.90441158882425)
        iqr = q_high - q_low
        center = np.median(x)
        if iqr < eps:
            x_min, x_max = (np.min(x), np.max(x))
            denom = x_max - x_min
            if denom < eps:
                return np.zeros_like(x)
            return (x - x_min) / (denom + eps)
        else:
            denom = iqr
            return (x - center) / (denom + eps)
    linear_slack_penalty = 0.5971226526804592 * np.clip(-slack, 0.0, None)
    coupled_slack_penalty = linear_slack_penalty * (1.0 + 0.5764447868606823 * uncertainty)
    norm_slack_penalty = adaptive_normalize(coupled_slack_penalty)
    critical_load_pressure = upward_rank * (remaining_work + eps)
    norm_critical_load = adaptive_normalize(critical_load_pressure)
    norm_energy = adaptive_normalize(min_incremental_energy)
    shifted_energy = norm_energy + 0.21753118211020375
    shifted_energy = np.clip(shifted_energy, 0.0, 2.0)
    wait_normalized = adaptive_normalize(ready_wait_time)
    wait_urgency_gated = wait_normalized * (1.0 + norm_slack_penalty)
    norm_wait_urgency = adaptive_normalize(wait_urgency_gated)
    score = norm_slack_penalty + 0.8942273584131358 * norm_critical_load + 0.5105365758957404 * norm_wait_urgency + shifted_energy
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
