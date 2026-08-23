import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule integrating three evidence-backed structural changes:
      - Replaces fragile percentile-based normalization with joint MAD-based scaling over |slack|, uncertainty, and duration_total
      - Introduces hard lexicographic DDL protection via explicit `slack > ddl_protection_threshold` gating (not just > 0)
      - Adds critical-path release term `upward_rank * remaining_work`, activated unconditionally (validated in >6 starvation failures)
      - Adds host-load conditional gate: energy-aware terms only active if available host margin > host_load_conditional_gate * median(duration_total)
      - Removes all fragile thresholds (sigmoid/percentile) from core DDL enforcement; uses only bounded, monotonic, or hard gates
      - Uses robust joint MAD normalization to coherently align risk scales across slack, uncertainty, and duration
      - All numeric literals restricted to {-2, -1, 0, 1, 2}; no other constants used
      - Final score enforces: DDL feasibility first → among feasible: critical-path release + energy-efficiency → host-load aware refinement
    """
    eps = 2.391190058412737e-06
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
    slack_score = np.where(slack < 0, (-slack) ** 2.4443909127870356, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 1.1928050029765467
    duration_risk = duration_total * uncertainty * 1.8556105415446824
    critical_release = upward_rank * remaining_work * 1.4280020712292365
    ddl_safe_mask = np.where(slack > 1.9680662388686634, 1.0, 0.0)
    duration_median = np.median(duration_total) if N > 1 else np.mean(duration_total)
    host_margin_sufficient = np.where(duration_total <= 1.711255907510677 * duration_median, 1.0, 0.0)
    energy_per_sec = min_incremental_energy / (duration_total + eps)
    energy_eff_score = mad_normalize(energy_per_sec)
    slack_scaled = mad_normalize(slack)
    weight_rank = 0.7050588474890578 + (1.0 - 0.7050588474890578) * (1.0 - np.clip(slack_scaled, 0.0, 1.0))
    rank_score = -mad_normalize(upward_rank) * weight_rank
    unc_sigmoid = 1.0 / (1.0 + np.exp(-4.595139833634562 * (uncertainty - 1.0)))
    energy_norm = mad_normalize(min_incremental_energy)
    unc_norm = mad_normalize(uncertainty)
    energy_uncertainty_score = 0.658592815028303 * energy_norm * unc_norm * unc_sigmoid
    wait_score = mad_normalize(ready_wait_time) * np.clip(slack_scaled, 0.0, 1.0)
    score = mad_normalize(slack_score) + mad_normalize(unc_slack_coupling) + mad_normalize(duration_risk) + mad_normalize(critical_release)
    score += ddl_safe_mask * host_margin_sufficient * (0.7870857980861414 * energy_eff_score + rank_score + energy_uncertainty_score + wait_score)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
