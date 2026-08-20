import numpy as np
RULE_METADATA = {'structure_hash': 'c7a4e9f5f6b888f0515a06ed1e2495b3f1ecc3115f51172a4e5c8229fc76166e', 'parameter_schema_hash': '063055cbdcb769daa9c7bd751f60020e6268d500423a334579f11063c3414a3e', 'best_parameter_hash': '127183b9fa541076c43f77ccc6beacf3acdab8405b9df96b212e0372ad76bc83', 'best_parameters': {'epsilon': 2.7783063363262928e-05, 'slack_penalty_exponent': 3.7535472167418296, 'criticality_scale': 1.7142562575154656, 'energy_sensitivity': 1.1970347377376085, 'uncertainty_gate_threshold': 0.39785116671333554, 'ddl_protection_gate_slope': 3.825759687592002, 'remaining_work_weight': 0.6580245179395927, 'energy_uncertainty_interaction': 0.5733307073054341, 'starvation_ramp_slope': 0.2297084485028327, 'rank_slack_hinge_offset': 0.7344033121834884}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': 'e658c1e72c011e93116252564370d67bf777af15a05c9da2db48c4af33e90f45', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: replaces steepened softplus starvation relief with *bounded linear ramp* for numerical safety, 
       monotonic discrimination, and zero-overhead at zero wait; reinstates clipped linear hinge for rank-slack coupling with tunable offset 
       to activate urgency earlier under mild deadline pressure; removes unused parameters (slack_pressure_gate_steepness, wait_decay) per validation;
       retains joint feasibility gate and median-MAD normalization; enforces strict [-2,2] clipping on all terms."""
    eps = 2.7783063363262928e-05
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
    joint_gate = 1.0 / (1.0 + np.exp(-3.825759687592002 * slack)) * 1.0 / (1.0 + np.exp(3.825759687592002 * (norm_uncert - 0.39785116671333554)))
    ddl_breach = (slack <= 0.0).astype(float)
    critical_path_leverage = norm_rank * norm_work * ddl_breach
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 3.7535472167418296
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    slack_hinge = np.clip(-norm_slack - 0.7344033121834884, 0.0, 1.0)
    boosted_rank = norm_rank * (1.0 + 1.7142562575154656 * slack_hinge)
    starvation_ramp = np.clip(0.2297084485028327 * (ready_wait_time + eps), 0.0, 2.0)
    wait_benefit = starvation_ramp
    energy_uncert_penalty = norm_energy * norm_uncert * joint_gate
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(critical_path_leverage, -2.0, 2.0) - np.clip(boosted_rank, -2.0, 2.0) - 1.1970347377376085 * np.clip(norm_energy * joint_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + 0.5733307073054341 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 0.6580245179395927 * np.clip(norm_work * joint_gate, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
