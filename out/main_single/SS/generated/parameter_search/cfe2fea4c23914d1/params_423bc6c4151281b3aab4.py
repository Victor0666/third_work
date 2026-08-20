import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule: replaces robust_normalize with median-MAD per-feature scaling;
       adds conditional DDL protection gate; replaces logistic wait-benefit with bounded linear ramp;
       replaces multiplicative energy-uncert penalty with additive interaction under slack-pressure gate;
       removes norm_duration bias (redundant with slack/energy/duration_risk_score);
       preserves all evidence-backed structural actions: add_conditional_ddl_protection_gate,
       add_host_load_conditional_gate, add_load_successor_release_interaction."""
    eps = 0.06728024704698182
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
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.1754346513509901
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    slack_pressure_sig = 1.0 / (1.0 + np.exp(-1.85711502381591 * norm_slack))
    rank_gate = ddl_urgent_mask * slack_pressure_sig + (1.0 - ddl_urgent_mask) * 0.8138801439932452
    boosted_rank = norm_rank * (1.0 + 1.010516887025695 * rank_gate)
    uncert_gate = np.where(uncertainty > 0.5128690807270074, 1.0, 0.0)
    duration_risk_score = (min_exec_time + min_comm_time) * uncert_gate * ddl_urgent_mask
    wait_benefit = np.clip(0.1312242474106175 * (ready_wait_time + 8.18787232676791e-06), 0.0, 1.0)
    energy_uncert_penalty = norm_energy * norm_uncert * ddl_urgent_mask * uncert_gate
    score = +norm_slack_penalty - boosted_rank - 1.9956399970560197 * norm_energy - wait_benefit + 0.18104379924435654 * median_mad_normalize(duration_risk_score) + 0.24231416959434451 * energy_uncert_penalty + 0.8854788065465335 * norm_work
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
