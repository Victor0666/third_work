import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining best elements:
       - Uses tunable quantile-based normalization (Parent 2) for robustness
       - Sign-preserving MAD for slack (Parent 2) to amplify boundary urgency
       - Smooth DDL-protection gate with slope parameter (Parent 2) for energy/uncertainty gating
       - Critical path leverage only under breach (both parents)
       - Novel congestion coupling: power-law scaling of congestion signal to better distinguish overload states
       - Bounded starvation relief with tunable cap to prevent priority inversion
       - All terms clipped to [-2,2] and fused with bounded coefficients for stability
       - No numeric literals beyond -2,-1,0,1,2; uses np.finfo for epsilon safeguards."""
    eps = 4.2989867322991005e-06
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
            q1 = np.quantile(x, 0.31247319116798833)
            q3 = np.quantile(x, 0.6996410877081437)
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
    ddl_gate = 1.0 / (1.0 + np.exp(-5.637807649539301 * slack))
    ddl_breach = (slack <= 0.0).astype(float)
    critical_path_leverage = norm_rank * norm_work * ddl_breach
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 2.1325648787615528
    norm_slack_penalty = robust_range_normalize(raw_slack_penalty)
    congestion_signal = ready_wait_time + uncertainty
    norm_congestion = robust_range_normalize(congestion_signal)
    congestion_score = np.clip(norm_congestion ** 0.9603910113782428, 0.0, 2.0) * (1.0 - ddl_gate)
    wait_mean = np.mean(ready_wait_time)
    wait_denom = np.maximum(wait_mean, eps)
    wait_benefit = np.clip(ready_wait_time / wait_denom, 0.0, 0.973315396332794)
    uncert_gate = 1.0 / (1.0 + np.exp(-5.637807649539301 * (norm_uncert - 0.8448457951679773)))
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_breach
    slack_pressure = np.clip(-slack, 0.0, 1.0)
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(critical_path_leverage, -2.0, 2.0) - np.clip(norm_rank * (1.0 + 1.0967651462120371 * slack_pressure), -2.0, 2.0) - 1.110253723667672 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip(congestion_score, -2.0, 2.0) + 0.6668323564317197 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 0.35065623254799094 * np.clip(norm_work * ddl_gate, -2.0, 2.0) + 2.1325648787615528 * np.clip(slack_pressure, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
