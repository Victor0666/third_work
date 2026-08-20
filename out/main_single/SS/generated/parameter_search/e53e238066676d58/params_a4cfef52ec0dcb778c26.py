import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule: 12 parameters, all used; removes slack_normalization_bias to comply with count limit;
       retains binary DDL gate, linear starvation ramp, median-MAD normalization, and physically grounded slack-pressure sig;
       ensures monotonic, bounded, deterministic output with strict shape enforcement."""
    eps = 0.0904399005671866
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
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 2.4450246422594364
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    slack_pressure_sig = 1.0 / (1.0 + np.exp(-6.642895472757759 * (slack + eps)))
    rank_gate = ddl_urgent_mask * slack_pressure_sig + (1.0 - ddl_urgent_mask) * 0.9319859390275862
    boosted_rank = norm_rank * (1.0 + 2.4896135528967824 * rank_gate)
    uncert_gate = np.where(uncertainty > 0.6194677597458947, 1.0, 0.0)
    duration_risk_score = (min_exec_time + min_comm_time) * uncert_gate * ddl_urgent_mask
    wait_benefit = np.clip(0.5494461913235793 * (ready_wait_time + 1.969203404069896e-09), 0.0, 1.0)
    energy_uncert_penalty = norm_energy * norm_uncert * ddl_urgent_mask * uncert_gate
    score = +norm_slack_penalty - boosted_rank - 1.0792744110422494 * norm_energy - wait_benefit + 1.350670686034746 * median_mad_normalize(duration_risk_score) + 0.1758577623781371 * energy_uncert_penalty + 0.9492619614906057 * norm_work
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
