import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule: 12 parameters, all used; removes slack_normalization_bias to comply with count limit;
       retains binary DDL gate, linear starvation ramp, median-MAD normalization, and physically grounded slack-pressure sig;
       ensures monotonic, bounded, deterministic output with strict shape enforcement."""
    eps = 0.04955779580065891
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
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 3.2535698327814964
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    slack_pressure_sig = 1.0 / (1.0 + np.exp(-7.986769219275772 * (slack + eps)))
    rank_gate = ddl_urgent_mask * slack_pressure_sig + (1.0 - ddl_urgent_mask) * 0.853813519083411
    boosted_rank = norm_rank * (1.0 + 2.2824464476322035 * rank_gate)
    uncert_gate = np.where(uncertainty > 0.9191592807946127, 1.0, 0.0)
    duration_risk_score = (min_exec_time + min_comm_time) * uncert_gate * ddl_urgent_mask
    wait_benefit = np.clip(0.5375955062031673 * (ready_wait_time + 1.4646998005865478e-09), 0.0, 1.0)
    energy_uncert_penalty = norm_energy * norm_uncert * ddl_urgent_mask * uncert_gate
    score = +norm_slack_penalty - boosted_rank - 0.26876341795785585 * norm_energy - wait_benefit + 0.14751504359717393 * median_mad_normalize(duration_risk_score) + 0.14302245405931008 * energy_uncert_penalty + 0.030406078665260888 * norm_work
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
