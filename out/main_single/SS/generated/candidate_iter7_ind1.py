import numpy as np
RULE_METADATA = {'structure_hash': 'cfe2fea4c23914d1ea31ea67457ecbd6b2a3d6097201b8fd4f96e743f368fa12', 'parameter_schema_hash': '82ae95a1860965af043489eadfa94cd42ee792eeb266a28a70298c068902510a', 'best_parameter_hash': 'd5327591142d0cb75a4ed82f8d02519f64da188f853f2ee8224f897fcd89efc9', 'best_parameters': {'epsilon': 0.005368192149437771, 'slack_penalty_exponent': 1.015702513155409, 'criticality_scale': 1.259710709866807, 'energy_sensitivity': 1.2089806244388779, 'duration_robustness': 0.5442747514355244, 'wait_decay': 0.0831002622480323, 'uncertainty_gate_threshold': 0.3640246062976861, 'slack_pressure_gate_steepness': 1.868155887488565, 'remaining_work_weight': 0.8879765380502185, 'wait_saturation_offset': 2.5185153209644584e-07, 'energy_uncertainty_interaction': 0.2276173798170245, 'ddl_urgent_default_gate': 0.9443241687439676}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': '560279d0d54302ac9d5911280b4849157f3b57c0356b9f0dd1b057095fe7f3c2', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule: replaces robust_normalize with median-MAD per-feature scaling;
       adds conditional DDL protection gate; replaces logistic wait-benefit with bounded linear ramp;
       replaces multiplicative energy-uncert penalty with additive interaction under slack-pressure gate;
       removes norm_duration bias (redundant with slack/energy/duration_risk_score);
       preserves all evidence-backed structural actions: add_conditional_ddl_protection_gate,
       add_host_load_conditional_gate, add_load_successor_release_interaction."""
    eps = 0.005368192149437771
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
        x = np.asarray(x)
        center = np.median(x)
        mad = np.median(np.abs(x - center))
        scale = mad + eps if N > 1 else np.abs(x[0] - center) + eps
        return (x - center) / scale
    norm_slack = median_mad_normalize(slack)
    norm_energy = median_mad_normalize(min_incremental_energy)
    norm_rank = median_mad_normalize(upward_rank)
    norm_work = median_mad_normalize(remaining_work)
    norm_wait = median_mad_normalize(ready_wait_time)
    norm_uncert = median_mad_normalize(uncertainty)
    ddl_urgent_mask = (slack <= 0.0).astype(float)
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.015702513155409
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    slack_pressure_sig = 1.0 / (1.0 + np.exp(-1.868155887488565 * norm_slack))
    rank_gate = ddl_urgent_mask * slack_pressure_sig + (1.0 - ddl_urgent_mask) * 0.9443241687439676
    boosted_rank = norm_rank * (1.0 + 1.259710709866807 * rank_gate)
    uncert_gate = np.where(uncertainty > 0.3640246062976861, 1.0, 0.0)
    duration_risk_score = (min_exec_time + min_comm_time) * uncert_gate * ddl_urgent_mask
    wait_benefit = np.clip(0.0831002622480323 * (ready_wait_time + 2.5185153209644584e-07), 0.0, 1.0)
    energy_uncert_penalty = norm_energy * norm_uncert * ddl_urgent_mask * uncert_gate
    score = +norm_slack_penalty - boosted_rank - 1.2089806244388779 * norm_energy - wait_benefit + 0.5442747514355244 * median_mad_normalize(duration_risk_score) + 0.2276173798170245 * energy_uncert_penalty + 0.8879765380502185 * norm_work
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
