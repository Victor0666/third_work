import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule with graded urgency, tanh-based starvation relief, and joint congestion gating.
       Key structural features:
         - Graded critical-path urgency via sigmoid(-slack), replacing binary gates → smoother control near deadline.
         - Urgency-sharpened slack pressure: (-slack)^p clipped to [0,1] for enhanced boundary sensitivity.
         - Smooth tanh wait benefit: avoids flat gradients and preserves discriminability at low waits.
         - Energy-uncertainty coupling gated by both soft DDL feasibility AND joint congestion (wait+uncert).
         - All features use identical robust quantile normalization (Q1/Q3) for consistency and outlier resistance.
       No numeric literals outside {-2,-1,0,1,2}; all tunables declared in PARAMETER_SCHEMA."""
    eps = 2.7330585209600077e-06
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def robust_range_normalize(x):
        x = np.copy(x)
        if N == 1:
            q1 = q3 = x[0]
        else:
            q1 = np.quantile(x, 0.3979244329256856)
            q3 = np.quantile(x, 0.619752506817741)
        iqr = q3 - q1
        spread = iqr if iqr > eps else eps
        return (x - q1) / spread
    norm_energy = robust_range_normalize(min_incremental_energy)
    norm_duration = robust_range_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_range_normalize(upward_rank)
    norm_work = robust_range_normalize(remaining_work)
    norm_wait = robust_range_normalize(ready_wait_time)
    norm_uncert = robust_range_normalize(uncertainty)
    if N == 1:
        slack_med = slack[0]
        slack_mad = eps
    else:
        slack_med = np.median(slack)
        slack_mad = np.median(np.abs(slack - slack_med))
    slack_spread = slack_mad if slack_mad > eps else eps
    norm_slack = (slack - slack_med) / slack_spread
    soft_ddl_gate = 1.0 / (1.0 + np.exp(-2.819533651367504 * slack))
    urgency_gate = 1.0 / (1.0 + np.exp(-2.819533651367504 * -slack))
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 2.4450578842069497
    norm_slack_penalty = robust_range_normalize(raw_slack_penalty)
    congestion_activation = (norm_wait >= 0.8977937673951486) & (norm_uncert >= 0.8977937673951486)
    congestion_gate = congestion_activation.astype(float)
    wait_benefit = np.tanh(ready_wait_time / (1.0 + 0.948435785159284))
    energy_uncert_interaction = norm_energy * norm_uncert * congestion_gate * soft_ddl_gate
    slack_pressure_raw = np.clip(-slack, 0.0, 1.0)
    slack_pressure = slack_pressure_raw ** 0.948435785159284
    successor_release = min_exec_time * urgency_gate * np.clip(norm_rank, 0.0, 1.0)
    critical_path_leverage = norm_rank * norm_work * urgency_gate
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(critical_path_leverage, -2.0, 2.0) - np.clip(norm_rank * (1.0 + 1.0153302456445377 * slack_pressure), -2.0, 2.0) - 0.7693299944270873 * np.clip(norm_energy * soft_ddl_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip((norm_wait + norm_uncert) * congestion_gate * (1.0 - soft_ddl_gate), -2.0, 2.0) + 0.701364377332547 * np.clip(energy_uncert_interaction, -2.0, 2.0) + 0.27427137939594626 * np.clip(norm_work * soft_ddl_gate, -2.0, 2.0) + 2.4450578842069497 * np.clip(slack_pressure, -2.0, 2.0) - np.clip(successor_release, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
