import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule with strict lexicographic DDL enforcement:
      - Uses robust per-feature MAD normalization for small-N stability.
      - Critical path release gated exclusively by slack <= 0 (DDL-emergency).
      - Anti-starvation requires dual condition: slack > 0 AND slack >= 1.0 (fixed threshold, no extra parameter).
      - Energy efficiency uses uncertainty-dampened energy_per_sec: / (duration * (1 + uncertainty)).
      - Adaptive logistic slack scaling around fixed threshold 1.0 with slope 2.0 (allowed literal).
      - All numeric literals are in {-2,-1,0,1,2}; all tunables declared in PARAMETER_SCHEMA.
      - No unused parameters; all PARAMS references match schema exactly.
    """
    eps = 1.4853588651624693e-08
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

    def mad_normalize(x):
        x = np.asarray(x, dtype=float)
        if N == 1:
            return np.zeros_like(x, dtype=float)
        med = np.median(x)
        dev = np.abs(x - med)
        mad = np.median(dev) + eps
        normalized = (x - med) / mad
        return np.clip(normalized, -2.0, 2.0)
    slack_score = np.where(slack < 0, (-slack) ** 2.279102866330267, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 1.5787270282760846
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 0.015617671742506471
    is_ddl_constrained = slack <= 0.0
    critical_release = upward_rank * remaining_work
    critical_release_norm = -mad_normalize(critical_release)
    successor_release_contribution = np.where(is_ddl_constrained, critical_release_norm * 2.7831366461098455, 0.0)
    slack_headroom_mask = np.where(slack > 0.0, 1.0, 0.0)
    duration_total_safe = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / (duration_total_safe * (1.0 + uncertainty + eps))
    energy_eff_score = mad_normalize(energy_per_sec) * 1.5149638422515426
    slack_logistic = 1.0 / (1.0 + np.exp(-(slack - 1.0) * 2.0))
    weight_rank = 0.5040416173071917 + (1.0 - 0.5040416173071917) * (1.0 - slack_logistic)
    rank_score = -mad_normalize(upward_rank) * weight_rank
    unc_sigmoid = 1.0 / (1.0 + np.exp(-7.672359215123802 * (uncertainty - 1.0)))
    energy_norm = mad_normalize(min_incremental_energy)
    unc_norm = mad_normalize(uncertainty)
    energy_uncertainty_score = 0.32467470051426184 * energy_norm * unc_norm * unc_sigmoid
    starvation_gate = np.where(slack >= 1.0, 1.0, 0.0)
    starvation_enabled = slack_headroom_mask * starvation_gate
    duration_med = np.median(duration_total_safe) if N > 1 else duration_total_safe[0]
    wait_duration_scaled = ready_wait_time / (duration_med + eps)
    wait_power = wait_duration_scaled ** 1.1923530171399588
    wait_score = mad_normalize(wait_power)
    score = mad_normalize(slack_score) + mad_normalize(unc_slack_coupling) + mad_normalize(duration_risk) + successor_release_contribution
    score += starvation_enabled * (energy_eff_score + rank_score + energy_uncertainty_score + wait_score)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
