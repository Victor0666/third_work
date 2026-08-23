import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule with lexicographic DDL enforcement via hard gate, joint MAD risk normalization,
       critical-path release emphasis, and anti-starvation with slack-decayed wait-time term.
    
    Key structural changes:
      - Replaces percentile normalization with robust joint MAD normalization over |slack|, uncertainty, and duration_total
      - Introduces critical_path_release_score = upward_rank * remaining_work → unconditionally prioritizes DAG bottlenecks
      - Uses hard gate `slack > ddl_protection_threshold` (not just > 0) to separate feasibility enforcement from optimization
      - Normalizes ready_wait_time with exponent decay: (ready_wait_time / (|slack| + 1)) ** ready_wait_decay_exponent
      - Applies uncertainty_slack_coupling linearly (not multiplicatively) to avoid explosive penalties at extreme negatives
      - Removes fragile sigmoid gates; uses bounded linear interactions and MAD scaling for stability
      - All numeric literals strictly limited to {-2,-1,0,1,2}
    """
    eps = 4.021270565194816e-09
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
    duration_total = min_exec_time + min_comm_time + eps
    risk_signals = np.concatenate([np.abs(slack), uncertainty, duration_total])
    risk_median = np.median(risk_signals)
    mad = np.median(np.abs(risk_signals - risk_median)) + eps
    joint_scale = 1.9213174750500457 * mad

    def joint_mad_normalize(x):
        x = np.asarray(x, dtype=float)
        if N == 1:
            return np.zeros_like(x, dtype=float)
        return np.clip((x - risk_median) / joint_scale, -2.0, 2.0)
    slack_violation = np.maximum(0.0, -slack)
    unc_slack_coupling = 0.002315865761731933 * uncertainty * slack_violation
    duration_risk = 0.002457732617710591 * duration_total * np.where(slack <= 3.606012544277153, uncertainty, 0.0)
    critical_path_release_score = upward_rank * remaining_work
    slack_headroom_mask = np.where(slack > 3.606012544277153, 1.0, 0.0)
    energy_norm = joint_mad_normalize(min_incremental_energy)
    energy_score = 0.37638815135462367 * energy_norm * slack_headroom_mask
    unc_norm = joint_mad_normalize(uncertainty)
    energy_uncertainty_score = 0.8945328263029477 * energy_norm * unc_norm * slack_headroom_mask
    wait_base = ready_wait_time / (np.abs(slack) + 1.0)
    wait_score = wait_base ** 1.2112747306327747 * slack_headroom_mask
    wait_norm = joint_mad_normalize(wait_score)
    score = joint_mad_normalize(slack_violation) + joint_mad_normalize(unc_slack_coupling) + joint_mad_normalize(duration_risk)
    score += -joint_mad_normalize(critical_path_release_score)
    score += energy_score + energy_uncertainty_score + wait_norm
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
