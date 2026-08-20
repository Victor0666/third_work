import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule addressing prior weaknesses:
       - Replaces raw duration-criticality with ROBUSTLY NORMALIZED (exec+comm) * rank, avoiding scale explosion.
       - Uses SIGMOID-based congestion gating (smooth, differentiable, bounded) instead of hard threshold → stable anti-starvation under load.
       - Eliminates unbounded slack_pressure_raw**exponent; replaces with sigmoidal urgency near zero slack.
       - Introduces 'duration_criticality_sigmoid' gate: soft transition from non-critical to critical based on signed_slack_norm.
       - All critical-path terms now use consistent robust normalization and bounded sigmoid gating.
       - Energy-uncertainty penalty only active under both congestion AND slack >= 0 (feasible + loaded).
    """
    eps = 6.0591170939657106e-05
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
            q_low = x[0]
            q_high = x[0]
            rng = eps
        else:
            q_low = np.quantile(x, 0.35215500608213)
            q_high = np.quantile(x, 0.8200833562974068)
            rng = q_high - q_low
        spread = rng if rng > eps else eps
        center = (q_low + q_high) / 2.0
        return (x - center) / spread
    norm_slack = robust_range_normalize(slack)
    norm_energy = robust_range_normalize(min_incremental_energy)
    norm_duration = robust_range_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_range_normalize(upward_rank)
    norm_work = robust_range_normalize(remaining_work)
    norm_wait = robust_range_normalize(ready_wait_time)
    norm_uncert = robust_range_normalize(uncertainty)
    ddl_feasible = (slack >= 0.0).astype(float)
    if N == 1:
        med_slack = slack[0]
        mad_slack = eps
    else:
        med_slack = np.median(slack)
        mad_slack = np.median(np.abs(slack - med_slack))
    spread_slack = mad_slack if mad_slack > eps else eps
    signed_slack_norm = (slack - med_slack) / spread_slack
    urgency_sigmoid = 1.0 / (1.0 + np.exp(-3.6226762221524638 * signed_slack_norm))
    duration_criticality = norm_duration * norm_rank * urgency_sigmoid
    ddl_breach = (slack <= 0.0).astype(float)
    critical_leverage = norm_rank * norm_work * ddl_breach
    clipped_rank = np.clip(norm_rank, 0.0, 2.0)
    successor_release = norm_duration * clipped_rank * ddl_breach
    joint_congestion_input = (norm_wait + norm_uncert) / 2.0
    congestion_gate = 1.0 / (1.0 + np.exp(-0.783959683450771 * (joint_congestion_input - 0.21241993647827967)))
    wait_benefit_base = 1.0 - np.exp(-0.2644780623447318 * (ready_wait_time + 6.483126610786664e-07))
    wait_benefit = wait_benefit_base * (1.0 - congestion_gate) * (1.0 + urgency_sigmoid)
    energy_uncert_penalty = norm_energy * norm_uncert * ddl_feasible * congestion_gate
    energy_slack_penalty = norm_energy * urgency_sigmoid * ddl_feasible
    score = +np.clip(signed_slack_norm, -2.0, 2.0) - np.clip(critical_leverage, -2.0, 2.0) - np.clip(successor_release, -2.0, 2.0) - np.clip(duration_criticality, -2.0, 2.0) - 0.6192699013192242 * np.clip(norm_energy * ddl_feasible, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + 0.5355376917376105 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 3.6226762221524638 * np.clip(energy_slack_penalty, -2.0, 2.0) + 0.9339166938026026 * np.clip(norm_work * ddl_feasible, -2.0, 2.0) + 0.7153406288006732 * np.clip(norm_rank * ddl_breach, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
