import numpy as np
RULE_METADATA = {'structure_hash': 'a63e283ec60a95357e90340e6400249ddde3ad3693447a26fbff1b6b55a5a8ea', 'parameter_schema_hash': '23893905a0c21bd4ee41553010f7a40e965e180cff1f7f23900c26da44a7833d', 'best_parameter_hash': '53a79065544c621c270301544c50b3a01a82ff1e119e3135e819cc15ab3f02b9', 'best_parameters': {'epsilon': 9.799392878192073e-05, 'slack_penalty_exponent': 2.1513363694259953, 'energy_sensitivity': 0.9754209891107913, 'criticality_scale': 1.1193272321390322, 'uncertainty_gate_threshold': 0.4883348444874798, 'slack_pressure_gate_steepness': 3.267114144982342, 'remaining_work_weight': 0.21174168200309165, 'wait_decay': 0.5491807058108825, 'energy_uncertainty_interaction': 0.7569457019020148, 'ddl_protection_gate_slope': 3.8140763651985736}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': '4c39733180d6c3cd07e4189f2b3fe4921ea1a2d33c2eda20bbe81e144894b8c6', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: reinstates validated critical_path_leverage with unified DDL-feasibility gating;
       replaces dual-gated energy with single robust smooth Heaviside (ddl_gate) — eliminates instability from linear ramp;
       introduces *normalized slack alignment* via clipped, signed slack deviation to decouple urgency from outlier magnitude;
       retains median-MAD normalization for all features; removes redundant duration_robustness & wait_saturation_offset;
       enforces strict monotonic slack pressure through bounded slack_alignment term;
       all intermediate terms clipped to [-2,2] for stability and AST depth control."""
    eps = 9.799392878192073e-05
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def median_mad_normalize(x):
        x = np.copy(x)
        if N == 1:
            med = x[0]
            mad = eps
        else:
            med = np.median(x)
            mad = np.median(np.abs(x - med))
        spread = mad if mad > eps else eps
        return (x - med) / spread
    norm_slack = median_mad_normalize(slack)
    norm_energy = median_mad_normalize(min_incremental_energy)
    norm_duration = median_mad_normalize(min_exec_time + min_comm_time)
    norm_rank = median_mad_normalize(upward_rank)
    norm_work = median_mad_normalize(remaining_work)
    norm_wait = median_mad_normalize(ready_wait_time)
    norm_uncert = median_mad_normalize(uncertainty)
    ddl_gate = 1.0 / (1.0 + np.exp(-3.8140763651985736 * slack))
    slack_alignment = np.clip(norm_slack, -1.0, 1.0)
    ddl_breach = (slack <= 0.0).astype(float)
    critical_path_leverage = norm_rank * norm_work * ddl_breach
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 2.1513363694259953
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-3.267114144982342 * (slack_pressure - 1.0)))
    boosted_rank = norm_rank * (1.0 + 1.1193272321390322 * slack_alignment) * rank_gate
    uncert_gate = 1.0 / (1.0 + np.exp(-3.8140763651985736 * (norm_uncert - 0.4883348444874798)))
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_gate
    wait_benefit = 1.0 - np.exp(-0.5491807058108825 * ready_wait_time)
    energy_slack_penalty = norm_energy * (1.0 + 2.1513363694259953 * slack_alignment) * ddl_gate
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(critical_path_leverage, -2.0, 2.0) - np.clip(boosted_rank, -2.0, 2.0) - 0.9754209891107913 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + 0.7569457019020148 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 2.1513363694259953 * np.clip(energy_slack_penalty, -2.0, 2.0) + 0.21174168200309165 * np.clip(norm_work, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
