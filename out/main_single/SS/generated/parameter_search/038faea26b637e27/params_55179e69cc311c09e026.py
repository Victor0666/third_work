import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule: combines Parent 2's numerical stability and DDL-protection gating with Parent 1's robust median-MAD normalization and explicit starvation relief offset.
       Novel improvements: (1) successor-release coupling raised to exponent for controlled amplification under deadline pressure;
       (2) linear slack penalty now scaled by learned coefficient instead of raw max(-slack,0); 
       (3) integrated anti-starvation via log-scaled wait time with small offset to prevent singularity — retains monotonicity while improving low-wait discrimination."""
    eps = 0.00012917754116664428
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def robust_normalize(x):
        x = np.asarray(x, dtype=float)
        if N == 1:
            med = x[0]
            mad = eps
        else:
            med = np.median(x)
            mad = np.median(np.abs(x - med))
        spread = mad + eps
        return (x - med) / spread
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    ddl_gate = np.clip(slack, 0.0, 1.0)
    ddl_gate = np.where(slack < 0.0, 0.0, ddl_gate)
    ddl_pressure = np.clip(-norm_slack, 0.0, 1.0)
    raw_slack_penalty = np.maximum(-slack, 0.0)
    norm_slack_penalty = robust_normalize(raw_slack_penalty)
    slack_penalty = 0.8591305192149403 * norm_slack_penalty
    successor_release = (norm_rank * norm_work) ** 1.2842521665329238 * ddl_pressure
    coupled_rank = norm_rank * (1.0 + 1.0959933165889544 * ddl_pressure)
    uncert_gate = np.where(norm_uncert > 0.14172697591887018, 1.0, 0.0)
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_gate
    wait_benefit = np.log1p(0.1671286695308898 * (ready_wait_time + eps))
    exec_penalty = norm_duration * ddl_pressure * ddl_gate
    score = +np.clip(slack_penalty, -2.0, 2.0) - np.clip(successor_release, -2.0, 2.0) - np.clip(coupled_rank, -2.0, 2.0) - 1.0414802150011497 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip(exec_penalty, -2.0, 2.0) + 0.4099731196634194 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 0.05925404089824927 * np.clip(norm_work, -2.0, 2.0) + np.clip(ddl_pressure * 3.292635469079549, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
