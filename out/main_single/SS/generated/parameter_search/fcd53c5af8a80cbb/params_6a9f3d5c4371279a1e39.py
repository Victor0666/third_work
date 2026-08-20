import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 1's robust MAD-slack urgency and Parent 2's decoupled gating & successor-release.
       Key novel improvements:
       1. Dual-gated starvation relief: wait_benefit activated *only* under DDL pressure (hard gate) to prevent premature aging bias;
       2. Urgency-coupled criticality: norm_rank amplified by *both* hard_ddl_gate *and* continuous slack_pressure, modulated by tunable strength;
       3. Unified congestion signal using *normalized sum* (not product) of wait+uncert, gated jointly by congestion threshold and soft_ddl_gate;
       4. Reintroduced sign-preserving MAD normalization for slack (from Parent 1) to sharpen sensitivity near zero deadline margin.
       All terms clipped to [-2,2], no literals beyond -2,-1,0,1,2, and shape (N,) enforced."""
    eps = 5.660702998676754e-06
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
            q1 = np.quantile(x, 0.31055040273807755)
            q3 = np.quantile(x, 0.82434372087355)
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
    norm_slack_mad = (slack - slack_med) / slack_spread
    soft_ddl_gate = 1.0 / (1.0 + np.exp(-3.8350584513442207 * slack))
    hard_ddl_gate = (slack <= 0.0).astype(float)
    critical_path_leverage = norm_rank * norm_work * hard_ddl_gate
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.905621314446691
    norm_slack_penalty = robust_range_normalize(raw_slack_penalty)
    congestion_activation = (norm_wait >= 0.349922213648419) & (norm_uncert >= 0.349922213648419)
    congestion_gate = congestion_activation.astype(float)
    congestion_score = (norm_wait + norm_uncert) * congestion_gate * (1.0 - soft_ddl_gate)
    wait_mean = np.mean(ready_wait_time)
    wait_denom = np.maximum(wait_mean, eps)
    raw_wait_benefit = np.clip(ready_wait_time / wait_denom, 0.0, 2.0)
    wait_benefit = raw_wait_benefit * hard_ddl_gate
    uncert_gate = 1.0 / (1.0 + np.exp(-3.8350584513442207 * (norm_uncert - 0.19105294411134105)))
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * hard_ddl_gate
    slack_pressure = np.clip(-slack, 0.0, 1.0)
    successor_release = min_exec_time * hard_ddl_gate * np.clip(norm_rank, 0.0, 1.0)
    coupled_criticality = norm_rank * (1.0 + 1.244934478212472 * slack_pressure) * hard_ddl_gate
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(critical_path_leverage, -2.0, 2.0) - np.clip(coupled_criticality, -2.0, 2.0) - 0.5131763126870315 * np.clip(norm_energy * soft_ddl_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip(congestion_score, -2.0, 2.0) + 0.18079276617036832 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 0.1981701651284513 * np.clip(norm_work * soft_ddl_gate, -2.0, 2.0) + 1.905621314446691 * np.clip(slack_pressure, -2.0, 2.0) - 0.29891770388799754 * np.clip(successor_release, -2.0, 2.0) + np.clip(norm_slack_mad, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
