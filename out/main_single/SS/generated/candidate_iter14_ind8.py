import numpy as np
RULE_METADATA = {'structure_hash': '038faea26b637e27c09c5c9939e28abd05d798973065c5f7b7b0ccfe9c7aaf05', 'parameter_schema_hash': '8db38eb6020ffdc73312532831f5711b84b8723750b76f2b581429bbe7cc34e7', 'best_parameter_hash': '37ced8f858caf0abe4f199389df940b829dbfa525e8bfbe88d1274e613cb65fe', 'best_parameters': {'epsilon': 0.015728288706885855, 'criticality_scale': 2.1014881677546624, 'energy_sensitivity': 1.0890319985356363, 'energy_uncertainty_interaction': 0.7271958035976486, 'remaining_work_weight': 0.3675628172717427, 'ddl_protection_gate_slope': 4.886513490359879, 'uncertainty_gate_threshold': 0.00010377923132635213, 'wait_decay': 0.3856618590313554, 'slack_penalty_linear_coeff': 2.0863278192805783, 'successor_release_exponent': 1.3511128123321836}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': '8bd1bc85a007c8440062fe9e44e8a978eccdd41df163678f68121fe2f7fbd432', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule: combines Parent 2's numerical stability and DDL-protection gating with Parent 1's robust median-MAD normalization and explicit starvation relief offset.
       Novel improvements: (1) successor-release coupling raised to exponent for controlled amplification under deadline pressure;
       (2) linear slack penalty now scaled by learned coefficient instead of raw max(-slack,0); 
       (3) integrated anti-starvation via log-scaled wait time with small offset to prevent singularity — retains monotonicity while improving low-wait discrimination."""
    eps = 0.015728288706885855
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
    slack_penalty = 2.0863278192805783 * norm_slack_penalty
    successor_release = (norm_rank * norm_work) ** 1.3511128123321836 * ddl_pressure
    coupled_rank = norm_rank * (1.0 + 2.1014881677546624 * ddl_pressure)
    uncert_gate = np.where(norm_uncert > 0.00010377923132635213, 1.0, 0.0)
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_gate
    wait_benefit = np.log1p(0.3856618590313554 * (ready_wait_time + eps))
    exec_penalty = norm_duration * ddl_pressure * ddl_gate
    score = +np.clip(slack_penalty, -2.0, 2.0) - np.clip(successor_release, -2.0, 2.0) - np.clip(coupled_rank, -2.0, 2.0) - 1.0890319985356363 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip(exec_penalty, -2.0, 2.0) + 0.7271958035976486 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 0.3675628172717427 * np.clip(norm_work, -2.0, 2.0) + np.clip(ddl_pressure * 4.886513490359879, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
