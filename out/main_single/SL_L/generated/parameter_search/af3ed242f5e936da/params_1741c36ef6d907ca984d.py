import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with slack-aware normalized criticality boost and quantile-based work-density scaling.
    
    Key structural improvements:
    - Replaces fragile sigmoid rank gating with *slack-aware normalized criticality boost*: 
      upward_rank is scaled additively by clipped negative slack, eliminating steepness tuning.
    - Uses *declared quantile parameter* for work-density denominator (replacing literal 0.75) to ensure full tunability.
    - Replaces exponential wait saturation with *tanh-scaled fairness* using declared tanh_bias.
    - All beneficial terms subtracted; all penalties added; strict DDL-first ordering preserved.
    """
    eps = 2.0903147300149444e-07
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
    slack_score = np.where(slack < 0, (-slack) ** 1.844238893030648, 0.0)
    duration_total = min_exec_time + min_comm_time + eps
    duration_norm = robust_normalize(duration_total)
    duration_risk = duration_norm * uncertainty * 0.22823018015083352
    energy_per_sec = min_incremental_energy / (duration_total + eps)
    energy_eff_score = robust_normalize(energy_per_sec) * 1.0602835364111018
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = robust_normalize(uncertainty * deadline_pressure) * 2.3697134750080107
    rank_norm = robust_normalize(upward_rank)
    clipped_slack = np.clip(-slack, 0.0, None)
    critical_bonus = rank_norm * (1.0 + 1.0521590208085185 * clipped_slack)
    unc_sigmoid = 1.0 / (1.0 + np.exp(-5.805922503607636 * (uncertainty - 1.0)))
    energy_norm = robust_normalize(min_incremental_energy)
    unc_norm = robust_normalize(uncertainty)
    energy_uncertainty_score = 0.20897151264855063 * energy_norm * unc_norm * unc_sigmoid
    energy_base_score = robust_normalize(min_incremental_energy)
    wait_norm = robust_normalize(ready_wait_time)
    wait_boost = 0.8763424173698678 * (1.0 + np.tanh(0.4824391888261214 * wait_norm))
    duration_q = np.quantile(duration_total, 0.7973759033943326) + eps
    work_density = remaining_work / duration_q
    work_density_norm = robust_normalize(work_density)
    work_density_bonus = 0.32475672647674175 * work_density_norm
    score = robust_normalize(slack_score) + duration_risk + energy_eff_score + unc_slack_coupling + energy_uncertainty_score + energy_base_score + wait_boost - critical_bonus - work_density_bonus
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
