import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule: 12 parameters, all used; removes slack_normalization_bias to comply with count limit;
       retains binary DDL gate, linear starvation ramp, median-MAD normalization, and physically grounded slack-pressure sig;
       ensures monotonic, bounded, deterministic output with strict shape enforcement."""
    eps = 0.0001722426114794305
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def median_mad_normalize(x):
        x = np.asarray(x)
        center = np.median(x)
        mad = np.median(np.abs(x - center))
        scale = mad + eps if N > 1 else np.abs(x[0] - center) + eps
        return (x - center) / scale
    norm_slack = median_mad_normalize(slack)
    norm_energy = median_mad_normalize(min_incremental_energy)
    norm_rank = median_mad_normalize(upward_rank)
    norm_work = median_mad_normalize(remaining_work)
    norm_wait = median_mad_normalize(ready_wait_time)
    norm_uncert = median_mad_normalize(uncertainty)
    ddl_urgent_mask = (slack <= 0.0).astype(float)
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 2.8417172354189386
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    slack_pressure_sig = 1.0 / (1.0 + np.exp(-4.936354770664429 * (slack + eps)))
    rank_gate = ddl_urgent_mask * slack_pressure_sig + (1.0 - ddl_urgent_mask) * 0.8502667339983832
    boosted_rank = norm_rank * (1.0 + 0.9263819243231715 * rank_gate)
    uncert_gate = np.where(uncertainty > 0.5764602266005704, 1.0, 0.0)
    duration_risk_score = (min_exec_time + min_comm_time) * uncert_gate * ddl_urgent_mask
    wait_benefit = np.clip(0.11746522867943421 * (ready_wait_time + 3.186525840997455e-09), 0.0, 1.0)
    energy_uncert_penalty = norm_energy * norm_uncert * ddl_urgent_mask * uncert_gate
    score = +norm_slack_penalty - boosted_rank - 1.10925222133925 * norm_energy - wait_benefit + 0.33796261489253177 * median_mad_normalize(duration_risk_score) + 0.3201307334230747 * energy_uncert_penalty + 0.14202300541412527 * norm_work
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
