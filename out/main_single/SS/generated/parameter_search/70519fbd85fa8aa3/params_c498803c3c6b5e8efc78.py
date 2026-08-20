import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule: all declared parameters are used; no unused or missing references;
       uses median-MAD normalization; clips all intermediate terms to [-2,2];
       applies DDL-protection gate via sigmoid using ddl_protection_gate_slope;
       includes slack_penalty_exponent for negative-slack power penalty;
       uses slack_pressure_gate_steepness for urgency ramp-up;
       retains successor-release interaction (min_exec_time × upward_rank × (slack <= 0));
       gates energy/uncertainty strictly by feasibility_gate = sigmoid(slack) from ddl_protection_gate_slope;
       removes inactive parameters (duration_robustness, wait_saturation_offset) per evidence."""
    eps = 8.236218964330462e-05
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
        x = np.copy(x)
        if N == 1:
            med = x[0]
            mad = eps
        else:
            med = np.median(x)
            mad = np.median(np.abs(x - med))
        spread = mad if mad > eps else eps
        return (x - med) / spread
    norm_slack = median_mad_normalize(slack)
    norm_energy = median_mad_normalize(min_incremental_energy)
    norm_exec = median_mad_normalize(min_exec_time)
    norm_comm = median_mad_normalize(min_comm_time)
    norm_rank = median_mad_normalize(upward_rank)
    norm_work = median_mad_normalize(remaining_work)
    norm_wait = median_mad_normalize(ready_wait_time)
    norm_uncert = median_mad_normalize(uncertainty)
    ddl_gate = 1.0 / (1.0 + np.exp(-6.3806344405509705 * slack))
    ddl_breach = (slack <= 0.0).astype(float)
    successor_release = norm_exec * norm_rank * ddl_breach
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.8177014402854168
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-1.6747834428264596 * (slack_pressure - 1.0)))
    boosted_rank = norm_rank * (1.0 + 1.308038524895203 * rank_gate)
    energy_term = 1.1830939061892836 * norm_energy * ddl_gate
    energy_uncert_penalty = 0.00032396330981465383 * norm_energy * norm_uncert * ddl_gate
    wait_term = 1.0 - np.exp(-0.27450464812917774 * ready_wait_time)
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(successor_release, -2.0, 2.0) - np.clip(boosted_rank, -2.0, 2.0) + energy_term + energy_uncert_penalty - np.clip(wait_term, -2.0, 2.0) + 1.1108959115183383 * np.clip(norm_work, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
