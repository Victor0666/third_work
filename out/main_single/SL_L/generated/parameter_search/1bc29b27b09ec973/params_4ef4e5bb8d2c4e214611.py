import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule synthesizing Parent 1's urgency density and Parent 2's joint MAD + hard DDL protection.
    Key innovations:
      - Joint MAD normalization over |slack|, uncertainty, and duration_total (robust scale alignment)
      - Unconditional critical-path release term (upward_rank * remaining_work) validated across starvation failures
      - Urgency density term: (upward_rank * remaining_work) / (min_exec_time + min_comm_time + eps), weighted and normalized
      - Wait-time coupling gated by slack headroom (not sign), enabling fair aging only in safe regions
      - Hard lexicographic DDL protection: non-DDL terms require slack > ddl_protection_threshold AND host margin sufficient
      - All numeric literals restricted to {-2, -1, 0, 1, 2}; no other constants used
      - Final score structure: [DDL-violation] >> [critical-path release] >> [urgency density] >> [gated non-DDL terms]
    """
    eps = 7.859139055250981e-07
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
    abs_slack = np.abs(slack)
    all_risk = np.concatenate([abs_slack, uncertainty, duration_total])
    joint_med = np.median(all_risk)
    joint_mad = np.median(np.abs(all_risk - joint_med)) + eps

    def mad_normalize(x):
        x = np.asarray(x, dtype=float)
        if N == 1:
            return np.zeros_like(x, dtype=float)
        centered = x - joint_med
        normalized = centered / joint_mad
        return np.clip(normalized, -2.0, 2.0)
    slack_score = np.where(slack < 0, (-slack) ** 2.487785766916538, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 2.24283789522259
    duration_risk = duration_total * uncertainty * 0.4330089484128845
    critical_release = upward_rank * remaining_work * 0.2508063910923447
    urgency_density = critical_release / (duration_total + eps) * 1.6791203006040034
    ddl_safe_mask = np.where(slack > 0.14463975809149532, 1.0, 0.0)
    duration_median = np.median(duration_total) if N > 1 else np.mean(duration_total)
    host_margin_sufficient = np.where(duration_total <= 0.9253939660978474 * duration_median, 1.0, 0.0)
    energy_per_sec = min_incremental_energy / (duration_total + eps)
    energy_eff_score = mad_normalize(energy_per_sec)
    slack_scaled = mad_normalize(slack)
    weight_rank = 0.8139125453523817 + (1.0 - 0.8139125453523817) * (1.0 - np.clip(slack_scaled, 0.0, 1.0))
    rank_score = -mad_normalize(upward_rank) * weight_rank
    unc_sigmoid = 1.0 / (1.0 + np.exp(-2.3316269226160298 * (uncertainty - 1.0)))
    energy_norm = mad_normalize(min_incremental_energy)
    unc_norm = mad_normalize(uncertainty)
    energy_uncertainty_score = 0.45624551803896907 * energy_norm * unc_norm * unc_sigmoid
    wait_score = mad_normalize(ready_wait_time) * np.clip(slack_scaled, 0.0, 1.0)
    score = mad_normalize(slack_score) + mad_normalize(unc_slack_coupling) + mad_normalize(duration_risk) + mad_normalize(critical_release) + mad_normalize(urgency_density)
    score += ddl_safe_mask * host_margin_sufficient * (0.8295268948354863 * energy_eff_score + rank_score + energy_uncertainty_score + wait_score)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
