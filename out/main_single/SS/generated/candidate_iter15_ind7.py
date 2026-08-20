import numpy as np
RULE_METADATA = {'structure_hash': 'e37e3f69ea1da9465fe7e9a36b1713ed5e847d49276c8efadc7347d8ca196266', 'parameter_schema_hash': '411c1ace1188b5197fc22e767516b26ab33b1a55c36b1b2787ae6cbfd3bce318', 'best_parameter_hash': '9077b57519f0ee0cb8caaf1a851847e2282a0fb3df83889678f9944fbc7c540c', 'best_parameters': {'epsilon': 2.8365270810382498e-05, 'criticality_scale': 1.6434905191601028, 'energy_sensitivity': 0.12340907469981827, 'energy_uncertainty_interaction': 0.2629991108126079, 'remaining_work_weight': 1.0661079147343147, 'ddl_protection_gate_slope': 5.662041122216956, 'uncertainty_gate_threshold': 0.7231512939214934, 'wait_decay': 0.5825855974222885, 'duration_robustness_weight': 0.9469980850913645}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': '8111af3e71fc3319d86dd34683fa5def26dd737b514c8b5cb04927e217c7e27e', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: replaces static duration penalty with uncertainty-weighted duration robustness;
       retains original clipped linear DDL gate (0→1 over [0,1]) for feasibility-first enforcement;
       reverts to linear successor-release interaction (norm_rank * norm_work * ddl_pressure) per consensus validation;
       adds explicit robustness term: penalizes tasks whose total duration (exec+comm) has high uncertainty *and* is large;
       all clipping bounds preserved at [-2,2] for AST depth control and numerical stability."""
    eps = 2.8365270810382498e-05
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
    total_duration = min_exec_time + min_comm_time
    norm_duration = median_mad_normalize(total_duration)
    norm_rank = median_mad_normalize(upward_rank)
    norm_work = median_mad_normalize(remaining_work)
    norm_wait = median_mad_normalize(ready_wait_time)
    norm_uncert = median_mad_normalize(uncertainty)
    ddl_gate = np.clip(slack, 0.0, 1.0)
    ddl_gate = np.where(slack < 0.0, 0.0, ddl_gate)
    ddl_pressure = np.clip(-norm_slack, 0.0, 1.0)
    successor_release = norm_rank * norm_work * ddl_pressure
    raw_slack_penalty = np.maximum(-slack, 0.0)
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    coupled_rank = norm_rank * (1.0 + 1.6434905191601028 * ddl_pressure)
    uncert_gate = np.where(norm_uncert > 0.7231512939214934, 1.0, 0.0)
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_gate
    wait_benefit = np.clip(0.5825855974222885 * ready_wait_time, 0.0, 1.0)
    robust_duration_penalty = norm_duration * norm_uncert * ddl_pressure * 0.9469980850913645
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(successor_release, -2.0, 2.0) - np.clip(coupled_rank, -2.0, 2.0) - 0.12340907469981827 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip(robust_duration_penalty, -2.0, 2.0) + 0.2629991108126079 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 1.0661079147343147 * np.clip(norm_work, -2.0, 2.0) + np.clip(ddl_pressure * 5.662041122216956, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
