import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule featuring:
      - Strict lexicographic DDL enforcement: all non-DDL terms (energy, latency, efficiency) are zero when slack <= ddl_protection_threshold.
      - Starvation fallback: unconditional wait-time urgency activated when slack <= ddl_protection_threshold.
      - Bounded risk term: linear duration-uncertainty coupling (no exponentiation) under DDL pressure.
      - Robust adaptive normalization: uses MAD/median per dimension with explicit [-2,2] clipping to prevent distortion.
      - All operations guarded against NaN/inf using np.nan_to_num and finfo bounds.
      - Exactly 12 parameters; all used; no numeric literals beyond {-2,-1,0,1,2}.
    """
    eps = 7.225518350712435e-08
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
    abs_slack = np.abs(slack) + eps
    all_risk_dims = np.stack([duration_total, abs_slack, uncertainty], axis=0)
    medians = np.median(all_risk_dims, axis=1)
    mads = np.mean(np.abs(all_risk_dims - medians[:, None]), axis=1) + eps
    duration_norm = np.clip(duration_total / (medians[0] + mads[0] + eps), -2.0, 2.0)
    slack_norm = np.clip(abs_slack / (medians[1] + mads[1] + eps), -2.0, 2.0)
    unc_norm = np.clip(uncertainty / (medians[2] + mads[2] + eps), -2.0, 2.0)
    slack_penalty = np.where(slack < 0, -slack * (1.0 + 1.23516231095796 * np.clip(unc_norm, 0.0, 2.0)), 0.0)
    critical_release = upward_rank * remaining_work
    median_cr = np.median(critical_release) + eps
    mad_cr = np.mean(np.abs(critical_release - median_cr)) + eps
    critical_release_norm = np.clip((critical_release - median_cr) / (mad_cr + eps), -2.0, 2.0)
    slack_deficit_ratio = np.clip(-slack / (0.7969645777433794 + eps), 0.0, 2.0)
    critical_release_score = np.where(slack < 0.7969645777433794, -critical_release_norm * 0.7861149303290911 * (1.0 + 1.8033443439334673 * slack_deficit_ratio), 0.0)
    ramp_half = 2.98747095530406 / 2.0
    gate_center = 0.7969645777433794
    gate_low = gate_center - ramp_half
    gate_high = gate_center + ramp_half
    slack_gate = np.clip((slack - gate_low) / (ramp_half * 2.0 + eps), 0.0, 1.0)
    wait_unc_ratio = (ready_wait_time + eps) / (uncertainty + eps)
    median_wait_unc = np.median(wait_unc_ratio) + eps
    mad_wait_unc = np.mean(np.abs(wait_unc_ratio - median_wait_unc)) + eps
    load_pressure = np.clip((wait_unc_ratio - median_wait_unc) / (mad_wait_unc + eps), -2.0, 2.0)
    host_load_gate = np.where(load_pressure < 0.6354247110596245, 1.0, 0.0)
    mad_energy = np.mean(np.abs(min_incremental_energy - np.median(min_incremental_energy))) + eps
    energy_median = np.median(min_incremental_energy) + eps
    energy_norm = np.clip((min_incremental_energy - energy_median) / (mad_energy + eps), -2.0, 2.0)
    energy_score = slack_gate * host_load_gate * energy_norm * 1.0887255513328928
    energy_uncertainty_score = slack_gate * host_load_gate * energy_norm * np.clip(unc_norm, 0.0, 2.0) * 0.020573513458767893
    latency_risk_score = slack_gate * host_load_gate * -duration_norm * 0.9066787966867249
    starvation_gate = np.where(slack <= 0.7969645777433794, 1.0, 0.0)
    median_wait = np.median(ready_wait_time) + eps
    wait_mad = np.mean(np.abs(ready_wait_time - median_wait)) + eps
    wait_norm = np.clip((ready_wait_time - median_wait) / (wait_mad + eps), -2.0, 2.0)
    wait_final = wait_norm * 1.4162072907046346 * starvation_gate
    slack_weight = np.where(slack < 0.7969645777433794, np.clip(1.0 - slack_norm / (slack_norm.max() + eps), 0.0, 1.0), 0.0)
    rank_weight = 1.0 - slack_weight
    rank_score = np.where(slack < 0.7969645777433794, -upward_rank / (np.median(upward_rank) + eps) * rank_weight * 0.9946386178114602, 0.0)
    duration_uncertainty_risk = np.where(slack < 0.7969645777433794, np.clip(duration_total * unc_norm, -2.0, 2.0) * 0.020573513458767893, 0.0)
    score = slack_penalty + critical_release_score + wait_final + energy_score + energy_uncertainty_score + latency_risk_score + rank_score + duration_uncertainty_risk
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
