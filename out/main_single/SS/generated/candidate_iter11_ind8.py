import numpy as np
RULE_METADATA = {'structure_hash': 'de2f106eacd6b63c8042a8af0818ffcc723c60245e8396ddd9123a1796e0d826', 'parameter_schema_hash': 'd53febc213a9513175c45688063a995a06d2048144d6031298afb3521003d2b7', 'best_parameter_hash': 'b8db49dad5e91c431ee1b8d8f43d01dba986e5c5b66423f2b4ed0182b96c4137', 'best_parameters': {'epsilon': 0.08969448862123587, 'slack_penalty_exponent': 2.6573593770564816, 'criticality_scale': 1.0955531768485098, 'energy_sensitivity': 0.948587518011783, 'duration_robustness': 0.42454868951902897, 'wait_decay': 0.14207142481883717, 'slack_pressure_gate_steepness': 5.283277367947896, 'remaining_work_weight': 0.569215536031231, 'wait_saturation_offset': 2.9164558114181333e-07, 'energy_uncertainty_interaction': 0.5645228568432302, 'slack_margin_ratio_threshold': 0.15132734934095687, 'energy_slack_modulation_slope': 0.6697961952565882}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': '68ca2e705296f4f5da47be421117bbd7524bf9da538c2e6744bd0d9a4c9396e5', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule: integrates Parent 2's DDL protection and successor-release logic with Parent 1's bounded rank-slack coupling and explicit starvation relief.
       Novel additions: (1) additive bias in rank-slack coupling replaced by unified sigmoid coupling, (2) linear energy modulation by slack pressure improves gradient continuity,
       (3) unified robust normalization with sign-preserving MAD scaling, (4) clipped composite terms to [-2,2] ensure ordinal stability under perturbation."""
    eps = 0.08969448862123587
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
        center = np.median(x) if N > 1 else x[0]
        spread = np.median(np.abs(x - center)) if N > 1 else np.abs(x[0] - center) + eps
        return (x - center) / (spread + eps)
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    slack_margin_ratio = np.abs(slack) / (min_exec_time + min_comm_time + eps)
    ddl_urgent = np.where((slack <= 0.0) | (slack_margin_ratio < 0.15132734934095687), 1.0, 0.0)
    successor_release_impact = norm_work * norm_rank
    slack_pressure_norm = np.clip(-norm_slack, 0.0, 2.0)
    energy_sensitivity_modulated = 0.948587518011783 * (1.0 + 0.6697961952565882 * slack_pressure_norm)
    rank_gate = 1.0 / (1.0 + np.exp(-5.283277367947896 * (slack_pressure_norm - 1.0)))
    boosted_rank = norm_rank * (1.0 + 1.0955531768485098 * rank_gate) * ddl_urgent
    wait_benefit = 1.0 - np.exp(-0.14207142481883717 * (norm_wait + 2.9164558114181333e-07))
    duration_risk_score = norm_duration * norm_uncert * ddl_urgent
    energy_uncert_penalty = norm_energy * norm_uncert * ddl_urgent
    score = +np.clip(norm_slack, -2.0, 2.0) + np.clip(norm_slack ** 2.6573593770564816, -2.0, 2.0) * ddl_urgent - np.clip(boosted_rank, -2.0, 2.0) - np.clip(successor_release_impact, -2.0, 2.0) - np.clip(norm_energy * energy_sensitivity_modulated * ddl_urgent, -2.0, 2.0) - np.clip(norm_duration * (1.0 - ddl_urgent), -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip(0.42454868951902897 * duration_risk_score, -2.0, 2.0) + np.clip(0.5645228568432302 * energy_uncert_penalty, -2.0, 2.0) + np.clip(0.569215536031231 * norm_work, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
