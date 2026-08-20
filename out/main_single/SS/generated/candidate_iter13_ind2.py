import numpy as np
RULE_METADATA = {'structure_hash': '41e31623faa1a3152da0f1b15f96e57d23fb39e921c4efd150bd4933a6db800d', 'parameter_schema_hash': '7946ca846457c3f64f013bb01d6336d94b2bef6ab7b53743bc5cafab43ddd428', 'best_parameter_hash': 'f52e906bd05e16299100ccd7f91d8a3ec30de18412b6bdcc85ab109beb9da5da', 'best_parameters': {'epsilon': 0.052616718067376576, 'slack_penalty_exponent': 3.146695381169035, 'criticality_scale': 0.6141019503313073, 'energy_sensitivity': 0.3317733734110574, 'energy_uncertainty_interaction': 0.46885355783620175, 'remaining_work_weight': 1.3804929288552388, 'ddl_protection_gate_slope': 4.381500569221039, 'uncertainty_gate_threshold': 0.5159399431065577, 'wait_decay': 0.4549162347293836, 'slack_margin_ratio_threshold': 0.03378223603304734}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': '84ec5ccfa54b08fcd898637f705407ce7f0bf592f84663e8e4c4f565966b84ad', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule: replaces literal 0.5 with PARAMS['slack_margin_ratio_threshold'];
       retains all high-confidence structural actions — conditional ddl protection, successor-release coupling, upward_rank × remaining_work;
       uses bounded linear slack scaling and median-MAD normalization;
       applies energy penalty only under safe+low-uncertainty conditions, and energy-uncertainty interaction only under medium-high uncertainty;
       eliminates redundant duration_robustness and wait_saturation_offset per evidence."""
    eps = 0.052616718067376576
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
    norm_rank = median_mad_normalize(upward_rank)
    norm_work = median_mad_normalize(remaining_work)
    norm_wait = median_mad_normalize(ready_wait_time)
    norm_uncert = median_mad_normalize(uncertainty)
    ddl_gate = 1.0 / (1.0 + np.exp(-4.381500569221039 * slack))
    abs_median_slack = np.abs(np.median(slack)) + eps
    slack_margin_ratio = np.clip(slack / abs_median_slack, -2.0, 2.0)
    ddl_urgent = (slack <= 0.0).astype(float)
    ddl_tight = (slack > 0.0) & (slack_margin_ratio < 0.03378223603304734)
    successor_release = norm_work * norm_rank * (ddl_urgent + ddl_tight.astype(float))
    clipped_slack = np.clip(norm_slack, -1.0, 1.0)
    slack_penalty = -clipped_slack
    slack_pressure = np.clip(1.0 - slack / abs_median_slack, 0.0, 1.0)
    urgency_boost = 1.0 + 0.6141019503313073 * slack_pressure
    critical_urgency = norm_rank * urgency_boost
    wait_benefit = 1.0 - np.exp(-0.4549162347293836 * ready_wait_time)
    low_uncert_gate = (norm_uncert < 0.5159399431065577).astype(float)
    energy_penalty = norm_energy * ddl_gate * low_uncert_gate
    high_uncert_gate = (norm_uncert >= 0.5159399431065577).astype(float)
    energy_uncert_interaction = norm_energy * norm_uncert * ddl_gate * high_uncert_gate
    score = +np.clip(slack_penalty, -2.0, 2.0) - np.clip(successor_release, -2.0, 2.0) - np.clip(critical_urgency, -2.0, 2.0) - 0.3317733734110574 * np.clip(energy_penalty, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + 0.46885355783620175 * np.clip(energy_uncert_interaction, -2.0, 2.0) + 1.3804929288552388 * np.clip(norm_work, -2.0, 2.0) + 3.146695381169035 * np.clip(-clipped_slack, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
