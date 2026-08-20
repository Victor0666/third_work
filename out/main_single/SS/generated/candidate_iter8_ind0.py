import numpy as np
RULE_METADATA = {'structure_hash': 'ba00c6ed2591521fa2d03fe1615c4fce8c87313f84a1ab4e918d5fcc1113f2e2', 'parameter_schema_hash': '2f1e560184bb420c664557b2986dbcf4df57bf20a309c20ae3428bdbc9fd534e', 'best_parameter_hash': '2ebc8cfb766ed82377327c85899acc5e284faf4f68b0e79c5f21a594108194ff', 'best_parameters': {'epsilon': 0.017430782683557498, 'ddl_urgency_boost': 970.2545583176635, 'criticality_scale': 1.8313780657574894, 'slack_pressure_gate_steepness': 7.440127228533459, 'energy_sensitivity': 1.152530173906047, 'duration_robustness': 0.2643885285947099, 'wait_decay': 0.16951127225845863, 'wait_saturation_offset': 1.827385031356659e-08, 'uncertainty_gate_threshold': 0.548136970147162, 'energy_uncertainty_interaction': 0.6413747684888575, 'remaining_work_weight': 1.097271772819184, 'negative_slack_only_activation': 0.9039844157907466}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': '2dce6c13ec3a801f0b01f0feebaa94bb33354cf2cabad116fac3b264b17a6f0d', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule: combines Parent 2's hard DDL gate and successor-release logic with Parent 1's negative-slack-only activation and robust median-MAD normalization.
       Introduces novel *dual-gated criticality*: upward_rank boosted only under slack<=0 (hard) AND via smooth sigmoid (soft), weighted by uncertainty.
       Uses unified robust normalization; removes unstable power penalties; enforces deterministic shape and finite output."""
    eps = 0.017430782683557498
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
        if N == 0:
            return np.zeros(0, dtype=float)
        center = np.median(x) if N > 1 else x[0]
        dev = x - center
        spread = np.median(np.abs(dev)) if N > 1 else np.abs(dev[0]) + eps
        return dev / (spread + eps)
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    ddl_urgent = (slack <= 0.0).astype(float)
    soft_slack_gate = 1.0 / (1.0 + np.exp(-7.440127228533459 * -norm_slack))
    uncertainty_mod = 1.0 + 0.9039844157907466 * norm_uncert * (ddl_urgent + (1.0 - ddl_urgent) * soft_slack_gate)
    protected_rank = norm_rank * (1.0 + 1.8313780657574894 * uncertainty_mod)
    successor_release_score = norm_work * norm_rank
    energy_uncert_penalty = norm_energy * np.where(norm_uncert > 0.548136970147162, 1.0, 0.0)
    negative_slack_mask = ddl_urgent * 0.9039844157907466
    duration_risk_score = norm_duration * norm_uncert * negative_slack_mask
    wait_benefit = 1.0 - np.exp(-0.16951127225845863 * (norm_wait + 1.827385031356659e-08))
    score = -ddl_urgent * 970.2545583176635 - protected_rank - successor_release_score - 1.152530173906047 * norm_energy - wait_benefit + 0.2643885285947099 * duration_risk_score + 0.6413747684888575 * energy_uncert_penalty + 1.097271772819184 * norm_work
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
