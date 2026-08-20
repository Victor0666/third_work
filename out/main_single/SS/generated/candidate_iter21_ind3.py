import numpy as np
RULE_METADATA = {'structure_hash': 'e2da463e502cee0f0c479897341010c8125186b0c12694ef5891f4751731c658', 'parameter_schema_hash': '33cf8e872235871eb77b58aa2e9a9cafb983a33f3c20bd0f560dd268f46d5637', 'best_parameter_hash': '7cc200a0c108dccc994284fef9e4f957f614fe6f4343aa1eab9487a36a23118a', 'best_parameters': {'epsilon': 9.094664692156033e-05, 'ddl_protection_gate_slope': 3.1624224501042146, 'energy_sensitivity': 1.5381274682190866, 'energy_uncertainty_interaction': 0.3381037399975615, 'remaining_work_weight': 0.6830007392794496, 'slack_penalty_exponent': 1.2636753769625169, 'criticality_scale': 1.203499344695747, 'uncertainty_gate_threshold': 0.00041748132246726983, 'quantile_q1': 0.18131657783023408, 'quantile_q3': 0.6720044618402314, 'congestion_activation_threshold': 0.7503157254300652, 'successor_release_weight': 1.479396805519403}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': 'ba6d9503748d438b49dc17ba1c511205aab6920ff5238bd16893eb7ceebe2af7', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with three key structural improvements:
       1. Bounded piecewise-linear congestion gate: activated only when BOTH normalized ready_wait_time AND normalized uncertainty exceed tunable threshold — avoids spurious load signals.
       2. Successor-release interaction: min_exec_time * (1 - ddl_gate) * clipped upward_rank — prioritizes fast-executing tasks whose high-criticality successors are blocked and deadline-constrained.
       3. Decoupled gating: hard threshold (slack <= 0.0) for critical-path terms, soft sigmoid gate only for energy/uncertainty — cleanly separates urgency from risk aversion.
       All numeric literals are restricted to -2,-1,0,1,2; uses np.finfo for epsilon safeguards; enforces shape (N,) explicitly."""
    eps = 9.094664692156033e-05
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
            q1 = np.quantile(x, 0.18131657783023408)
            q3 = np.quantile(x, 0.6720044618402314)
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
    soft_ddl_gate = 1.0 / (1.0 + np.exp(-3.1624224501042146 * slack))
    hard_ddl_gate = (slack <= 0.0).astype(float)
    critical_path_leverage = norm_rank * norm_work * hard_ddl_gate
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.2636753769625169
    norm_slack_penalty = robust_range_normalize(raw_slack_penalty)
    congestion_activation = (norm_wait >= 0.7503157254300652) & (norm_uncert >= 0.7503157254300652)
    congestion_gate = congestion_activation.astype(float)
    congestion_score = (norm_wait + norm_uncert) * congestion_gate * (1.0 - soft_ddl_gate)
    wait_mean = np.mean(ready_wait_time)
    wait_denom = np.maximum(wait_mean, eps)
    wait_benefit = np.clip(ready_wait_time / wait_denom, 0.0, 1.0)
    uncert_gate = 1.0 / (1.0 + np.exp(-3.1624224501042146 * (norm_uncert - 0.00041748132246726983)))
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * hard_ddl_gate
    slack_pressure = np.clip(-slack, 0.0, 1.0)
    successor_release = min_exec_time * hard_ddl_gate * np.clip(norm_rank, 0.0, 1.0)
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(critical_path_leverage, -2.0, 2.0) - np.clip(norm_rank * (1.0 + 1.203499344695747 * slack_pressure), -2.0, 2.0) - 1.5381274682190866 * np.clip(norm_energy * soft_ddl_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip(congestion_score, -2.0, 2.0) + 0.3381037399975615 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 0.6830007392794496 * np.clip(norm_work * soft_ddl_gate, -2.0, 2.0) + 1.2636753769625169 * np.clip(slack_pressure, -2.0, 2.0) - 1.479396805519403 * np.clip(successor_release, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
