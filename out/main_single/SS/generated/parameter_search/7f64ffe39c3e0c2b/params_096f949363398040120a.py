import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule: combines Parent 2's hard DDL gate and linear slack pressure with novel duration exponentiation
       and refined slack pressure width control. Introduces power-law duration penalty (exponent > 1) for stronger differentiation
       among late-critical tasks while preserving linearity for non-critical ones. Slack pressure now uses tunable width
       instead of fixed 1s ramp, enabling adaptive urgency coupling. Host-load proxy remains synthesized from wait+uncert,
       and all energy terms remain strictly gated by hard_ddl_gate — ensuring feasibility-first ranking. Normalization
       uses median-MAD for robustness; all outputs clipped to [-2,2] and safeguarded against inf/nan."""
    eps = 0.08647242070361198
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
    norm_duration = median_mad_normalize(min_exec_time + min_comm_time)
    norm_rank = median_mad_normalize(upward_rank)
    norm_work = median_mad_normalize(remaining_work)
    norm_wait = median_mad_normalize(ready_wait_time)
    norm_uncert = median_mad_normalize(uncertainty)
    load_proxy = 1.0 / (1.0 + np.exp(-2.0 * (norm_wait + norm_uncert)))
    hard_ddl_gate = np.where(slack < 0.0, 0.0, 1.0)
    slack_pressure = np.clip(-slack / (1.960833407454239 + eps), 0.0, 1.0)
    successor_release = norm_rank * norm_work * slack_pressure
    raw_slack_penalty = np.maximum(-slack, 0.0)
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    coupled_rank = norm_rank * (1.0 + 2.3583651052578474 * slack_pressure)
    uncert_gate = np.where(norm_uncert > 0.4068255710949378, 1.0, 0.0)
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * hard_ddl_gate
    wait_benefit = np.clip(0.8851569539770306 * ready_wait_time, 0.0, 1.0)
    duration_base = np.abs(norm_duration) * slack_pressure * hard_ddl_gate
    exec_penalty = np.power(np.abs(duration_base) + eps, 1.1526812701316278)
    load_gate = np.where(load_proxy > 0.9515562853514069, 1.0, 0.0)
    load_scaled_energy = norm_energy * (1.0 + 0.40010495685690234 * load_gate)
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(successor_release, -2.0, 2.0) - np.clip(coupled_rank, -2.0, 2.0) - 0.47681992962893716 * np.clip(load_scaled_energy * hard_ddl_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip(exec_penalty, -2.0, 2.0) + 0.7668830461021933 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 1.3996772885513404 * np.clip(norm_work, -2.0, 2.0) + np.clip(slack_pressure * 2.2887068888690583, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
