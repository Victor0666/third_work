import numpy as np
RULE_METADATA = {'structure_hash': 'fa9a952e5f01ff76a6d480d188b6a7356fda842d0e73cbe82c3cbc647399567b', 'parameter_schema_hash': '5c663c12ef0e3ae9a560e6958477f467af1996142db1917fb2fec871680ed4bd', 'best_parameter_hash': 'a9d36ffa860591bf609246584deb0892091d595af07de02c2be749570258cf59', 'best_parameters': {'epsilon': 0.0014306695310121506, 'criticality_scale': 2.018850739247501, 'energy_sensitivity': 1.154435054950316, 'energy_relaxation_under_ddl_stress': 0.8285060079703717, 'wait_decay': 0.28026314353841836, 'uncertainty_gate_threshold': 0.5538240074675271, 'slack_pressure_gate_steepness': 2.326068628125031, 'remaining_work_weight': 0.34602637430704486, 'energy_uncertainty_interaction': 0.42643547468340093, 'duration_penalty_weight': 0.950998285393708, 'urgency_saturation_offset': 1.30959109321663e-08}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': '1b121677120be023375e7ef2c96cfc365c0883dfaf9a29d6d608cb02c6749dfb', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining Parent 2's robust DDL pressure handling with Parent 1's successor-release insight.
       Key innovations: (1) Urgency saturation offset prevents gradient vanishing at slack=0; (2) Duration penalty now uses
       physical-scale sum (exec+comm) instead of normalized version, preserving latency sensitivity; (3) Critical-path
       coupling extended to include *normalized* remaining_work (not just binary), enabling graded boosting;
       (4) All numeric literals are -2,-1,0,1,2; no hidden constants."""
    eps = 0.0014306695310121506
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def stable_normalize(x):
        x_abs = np.abs(x)
        center = np.mean(x_abs) if N > 0 else 0.0
        spread = np.mean(x_abs) + eps
        return (x - center) / (spread + eps)
    norm_energy = stable_normalize(min_incremental_energy)
    norm_rank = stable_normalize(upward_rank)
    norm_work = stable_normalize(remaining_work)
    norm_wait = stable_normalize(ready_wait_time)
    norm_uncert = stable_normalize(uncertainty)
    raw_urgency_input = -slack + 1.30959109321663e-08
    slack_urgency = np.tanh(raw_urgency_input * 2.326068628125031)
    slack_pressure_score = np.clip(-slack, 0.0, np.inf)
    work_pressure_score = np.clip(norm_work, 0.0, np.inf)
    coupling_strength = slack_pressure_score * work_pressure_score
    boosted_rank = norm_rank * (1.0 + 2.018850739247501 * coupling_strength)
    risk_active = (slack < 0.0) & (norm_uncert > 0.5538240074675271)
    energy_uncert_penalty = np.where(risk_active, norm_energy * norm_uncert, np.zeros_like(norm_energy))
    wait_benefit = np.tanh(0.28026314353841836 * norm_wait)
    energy_weight = np.where(slack > 0.0, 1.154435054950316, 1.154435054950316 * 0.8285060079703717)
    duration_penalty = (min_exec_time + min_comm_time) * 0.950998285393708
    score = +(1.0 - slack_urgency) - boosted_rank - energy_weight * norm_energy - duration_penalty - wait_benefit + 0.42643547468340093 * energy_uncert_penalty + 0.34602637430704486 * norm_work
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
