import numpy as np
RULE_METADATA = {'structure_hash': '5fb75c2c853319500b8829d48e845bd4f551be06e9b273e4ebf1d7565c405eb4', 'parameter_schema_hash': '8ba4dadccdb6a705d2ebd9f85a5fd924072e6f806966197f9fc5c80d5edbb57b', 'best_parameter_hash': '77db9dacf8acc7a8f26c0dc2d82973f91e4eeb435590f1c65120dcb9fefa0b46', 'best_parameters': {'ddl_protection_gate_slope': 9.994939713919807, 'energy_sensitivity': 0.6833395182647879, 'energy_uncertainty_interaction': 0.21652857280467014, 'criticality_scale': 1.0897306930157658, 'remaining_work_weight': 1.2739293266119514, 'slack_penalty_exponent': 1.0035692205549944, 'slack_pressure_gate_steepness': 3.103901357800711, 'uncertainty_gate_threshold': 0.22695177016683374, 'wait_decay': 0.6098293429787934, 'rank_amplifier_lower_bound': 0.4762166903775267, 'rank_amplifier_upper_bound': 1.044110938063051, 'exec_comm_separation_factor': 0.9831117233102209}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': '2d58cbac7a62a8bda77a710f4a38db2ce9f70328414efe7177b7107b1efbdcbc', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: removes aggregated duration penalty to restore discriminative power;
       introduces *exec_comm_separation_factor* to independently gate and scale execution vs communication penalties;
       preserves all successful normalization, gating, and slack-coupling from Parent 2;
       ensures compute-bound tasks get urgent scheduling under tight slack while I/O-bound tasks are deprioritized only if bandwidth-limited;
       avoids structural overloading by decoupling exec/comm signals instead of summing them."""
    eps = np.finfo(float).tiny
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
        x = np.copy(x)
        if N == 1:
            center = x[0]
            spread = eps
        else:
            center = np.mean(x)
            spread = np.std(x, ddof=0)
        spread = np.where(spread > eps, spread, eps)
        return (x - center) / spread
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_exec = robust_normalize(min_exec_time)
    norm_comm = robust_normalize(min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    ddl_gate = 1.0 / (1.0 + np.exp(-9.994939713919807 * slack))
    raw_slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    slack_pressure_gate = 1.0 / (1.0 + np.exp(-3.103901357800711 * (raw_slack_pressure - 1.0)))
    ddl_breach = (slack <= 0.0).astype(float)
    critical_path_score = norm_rank * norm_work * ddl_breach
    rank_amplifier = 1.0 + 1.0897306930157658 * slack_pressure_gate
    rank_amplifier = np.clip(rank_amplifier, 0.4762166903775267, 1.044110938063051)
    boosted_rank = norm_rank * rank_amplifier
    uncert_gate = 1.0 / (1.0 + np.exp(-9.994939713919807 * (norm_uncert - 0.22695177016683374)))
    comm_uncert_penalty = norm_comm * norm_uncert * uncert_gate * ddl_breach
    wait_bonus = 0.6098293429787934 * norm_wait
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_gate
    energy_slack_penalty = norm_energy * raw_slack_pressure * ddl_gate
    exec_penalty = 0.9831117233102209 * norm_exec * slack_pressure_gate * ddl_gate
    comm_penalty = (1.0 - 0.9831117233102209) * norm_comm * slack_pressure_gate * ddl_gate
    score = +np.clip(norm_slack, -2.0, 2.0) + np.clip(raw_slack_pressure * 1.0035692205549944, -2.0, 2.0) - np.clip(critical_path_score, -2.0, 2.0) - np.clip(boosted_rank, -2.0, 2.0) - 0.6833395182647879 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) + np.clip(wait_bonus, -2.0, 2.0) + 0.21652857280467014 * np.clip(energy_uncert_penalty, -2.0, 2.0) + np.clip(comm_uncert_penalty, -2.0, 2.0) + 1.2739293266119514 * np.clip(norm_work, -2.0, 2.0) + np.clip(exec_penalty, -2.0, 2.0) + np.clip(comm_penalty, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
