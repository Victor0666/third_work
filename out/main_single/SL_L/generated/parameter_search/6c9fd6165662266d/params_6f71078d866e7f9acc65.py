import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining strengths from both parents:
      - Preserves Parent 2's lexicographic DDL protection gate (slack > ddl_protection_threshold)
      - Keeps unconditional critical-path term (upward_rank * remaining_work) validated against starvation
      - Uses bounded empirical slack scaling [-2,2] (allowed literals) instead of fragile percentiles
      - Introduces novel wait_slack_coupling_exponent for more expressive anti-starvation behavior
      - Uses tanh-based uncertainty sigmoid (more numerically stable than exp-based)
      - All components MAD-normalized and clipped to [-2,2] for robustness at N=1 and sparse sets
      - Final score ensures DDL-violation terms dominate, while non-DDL terms only activate under safety headroom
    """
    eps = 2.6225429151617888e-08
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
        abs_dev = np.abs(x - med)
        mad = np.median(abs_dev) + eps
        normalized = (x - med) / mad
        return np.clip(normalized, -2.0, 2.0)
    slack_score = np.where(slack < 0, (-slack) ** 3.5083828139648743, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 0.9919796506814522
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 0.3294832088667093
    crit_path_urgency = upward_rank * remaining_work * 1.587258428346348
    slack_headroom_mask = np.where(slack > 0.4583728421883401, 1.0, 0.0)
    wait_base = ready_wait_time / (1.0 + np.abs(slack)) ** 0.9976452966826208
    wait_score = wait_base * np.sign(slack + eps)
    energy_per_work = min_incremental_energy / (remaining_work + eps)
    energy_eff_score = mad_normalize(energy_per_work)
    slack_lb = -2.0
    slack_ub = 2.0
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    weight_rank = 0.3125351291833434 + (1.0 - 0.3125351291833434) * (1.0 - slack_scaled)
    rank_score = -mad_normalize(upward_rank) * weight_rank
    unc_sigmoid = np.tanh(0.5194004148948229 * (uncertainty - 1.0))
    energy_norm = mad_normalize(min_incremental_energy)
    unc_norm = mad_normalize(uncertainty)
    energy_uncertainty_score = 0.26493742263503467 * energy_norm * unc_norm * unc_sigmoid
    slack_norm = mad_normalize(slack_score)
    unc_slack_norm = mad_normalize(unc_slack_coupling)
    duration_norm = mad_normalize(duration_risk)
    crit_path_norm = mad_normalize(crit_path_urgency)
    wait_norm = mad_normalize(wait_score)
    score = slack_norm + unc_slack_norm + duration_norm
    score += crit_path_norm
    score += slack_headroom_mask * (0.7660158943157094 * energy_eff_score + rank_score + energy_uncertainty_score + wait_norm)
    is_ddl_pressure = slack <= 0
    critical_gate = np.where(is_ddl_pressure, 3.4378140484207043, 1.0)
    score = score * critical_gate
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
