import numpy as np
RULE_METADATA = {'structure_hash': '7fc36187624bfae2009a8f0ebeee2ef3f0d0079c869d424e8a1c45b5c0171598', 'parameter_schema_hash': '82586bd15aa64171f56867f150cede92f73b84e2fa41f332ebe855585d6a2823', 'best_parameter_hash': '77628a60e7d808fd4c100958a228b830fa6f8c8997273e963b63fcb7b033b833', 'best_parameters': {'epsilon': 4.7646883519846034e-05, 'criticality_scale': 1.4933997340702634, 'energy_sensitivity': 0.879327045291096, 'energy_uncertainty_interaction': 0.3283230182075538, 'remaining_work_weight': 0.9935653135764865, 'wait_saturation_offset': 0.0009939867269975742, 'quantile_low': 0.3668542019727982, 'quantile_high': 0.6962395894203877}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': 'b8ffd70ea52107bca109062e893700884b3d4bfc200096726f199d99c4fcb4e8', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule: uses robust quantile-based normalization;
       introduces successor-release interaction (min_exec_time × clipped_upward_rank × (slack <= 0));
       enforces hard DDL feasibility via binary gating;
       retains only validated structural terms;
       uses sign-preserving linear slack urgency and bounded wait bonus."""
    eps = 4.7646883519846034e-05
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def robust_quantile_normalize(x):
        x = np.copy(x)
        if N == 1:
            q_low = q_high = x[0]
        else:
            q_low = np.quantile(x, 0.3668542019727982)
            q_high = np.quantile(x, 0.6962395894203877)
        iqr = q_high - q_low
        spread = iqr if iqr > eps else eps
        return (x - q_low) / spread
    norm_slack = robust_quantile_normalize(slack)
    norm_energy = robust_quantile_normalize(min_incremental_energy)
    norm_exec = robust_quantile_normalize(min_exec_time)
    norm_rank = robust_quantile_normalize(upward_rank)
    norm_work = robust_quantile_normalize(remaining_work)
    norm_wait = robust_quantile_normalize(ready_wait_time)
    norm_uncert = robust_quantile_normalize(uncertainty)
    ddl_breach = (slack <= 0.0).astype(float)
    clipped_rank = np.clip(norm_rank, 0.0, 2.0)
    successor_release = norm_exec * clipped_rank * ddl_breach
    critical_path_importance = norm_rank * norm_work * ddl_breach
    raw_slack_urgency = -np.clip(slack, -np.inf, 0.0)
    norm_slack_urgency = robust_quantile_normalize(raw_slack_urgency)
    slack_penalty = np.clip(norm_slack_urgency, -2.0, 2.0)
    wait_threshold = np.quantile(ready_wait_time, 0.6962395894203877)
    uncert_threshold = np.quantile(uncertainty, 0.6962395894203877)
    wait_high = (ready_wait_time >= wait_threshold).astype(float)
    uncert_high = (uncertainty >= uncert_threshold).astype(float)
    congestion_gate = wait_high * uncert_high
    energy_feasible = (slack >= 0.0).astype(float)
    energy_penalty = norm_energy * energy_feasible
    energy_uncert_coupling = norm_energy * norm_uncert * congestion_gate * energy_feasible
    wait_bonus = np.clip(ready_wait_time / (ready_wait_time + 0.0009939867269975742), 0.0, 1.0)
    score = +slack_penalty - 1.4933997340702634 * critical_path_importance - 1.4933997340702634 * successor_release - 0.879327045291096 * energy_penalty - 0.3283230182075538 * energy_uncert_coupling - wait_bonus + 0.9935653135764865 * norm_work
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
