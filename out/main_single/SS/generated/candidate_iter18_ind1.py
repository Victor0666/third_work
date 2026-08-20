import numpy as np
RULE_METADATA = {'structure_hash': '3620937136cc6f011028f277fd28a96614418e0d2a03b9b6311419fa89b742a6', 'parameter_schema_hash': '3a5cdbf3f3e1efffcc0b32e392cdf2386b09b0fb2ed0dca3fdeab3a2cbfef941', 'best_parameter_hash': 'e05615a524a2ec642148102c5db706688eb60749404ebd336feebed78dc0cd9d', 'best_parameters': {'epsilon': 1.436078512578516e-06, 'criticality_scale': 2.153662752513309, 'energy_sensitivity': 0.44625558119672426, 'energy_uncertainty_interaction': 0.2092939473289015, 'remaining_work_weight': 1.9009767745247619, 'ddl_protection_gate_slope': 2.498905127689679, 'uncertainty_gate_threshold': 0.594077208158392, 'wait_decay': 0.7924171094529042, 'slack_penalty_linear_coeff': 1.3274813629166455, 'duration_robustness_factor': 1.1466017585368293, 'slack_energy_suppression_strength': 0.3356404208521301}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': '7b786577c77045e49d4b1d3018bafde0d667c4dbc381b5dce8a3201e2ecbf8e3', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: restores strict binary slack-driven energy suppression (per reflection), replaces log wait with clipped linear, removes graded rank-uncert coupling.
       Key structural improvement: introduces `slack_energy_suppression_strength` to control hardness of energy suppression — enables tunable tradeoff between DDL enforcement and energy minimization in marginal feasibility.
       All energy-related terms now use unified hard gate: `np.where(slack < 0, 0.0, 1.0)` instead of smooth ddl_gate — eliminates leakage risk under deadline violation.
       Starvation relief reverts to simple clipped linear form for robust low-wait discriminability."""
    eps = 1.436078512578516e-06
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
    hard_feasibility_gate = np.where(slack < 0.0, 0.0, 1.0)
    ddl_pressure = np.clip(-norm_slack, 0.0, 1.0)
    raw_slack_penalty = np.maximum(-slack, 0.0)
    norm_slack_penalty = robust_normalize(raw_slack_penalty)
    slack_penalty = 1.3274813629166455 * norm_slack_penalty
    successor_release = norm_rank * norm_work * ddl_pressure
    coupled_rank = norm_rank * (1.0 + 2.153662752513309 * ddl_pressure)
    uncert_gate = np.where(norm_uncert > 0.594077208158392, 1.0, 0.0)
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * hard_feasibility_gate
    wait_benefit = np.clip(0.7924171094529042 * ready_wait_time, 0.0, 2.0)
    duration_penalty = 1.1466017585368293 * norm_duration * ddl_pressure * hard_feasibility_gate
    slack_energy_suppress = np.where(slack < 0.0, 1.0 - 0.3356404208521301, 1.0)
    suppressed_norm_energy = norm_energy * slack_energy_suppress
    score = +np.clip(slack_penalty, -2.0, 2.0) - np.clip(successor_release, -2.0, 2.0) - np.clip(coupled_rank, -2.0, 2.0) - 0.44625558119672426 * np.clip(suppressed_norm_energy, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip(duration_penalty, -2.0, 2.0) + 0.2092939473289015 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 1.9009767745247619 * np.clip(norm_work, -2.0, 2.0) + np.clip(ddl_pressure * 2.498905127689679, -2.0, 2.0)
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.reshape(-1)
