import numpy as np
RULE_METADATA = {'structure_hash': '21255bcb530ba8b15da874b86c690fdfd2942588e0da7ee771ddb3aceea4fc76', 'parameter_schema_hash': '64d156f959fa5d94ade6ed7d8bfc20845ae32d029cef2d7f694cd7acaa3941d5', 'best_parameter_hash': '1f4f8e80afff738a8a2b30d9200fe2895a5440ea093efba8c9de45429f393d71', 'best_parameters': {'epsilon': 2.1134145571677787e-06, 'ddl_protection_gate_threshold': 0.1882548906640742, 'upward_rank_remaining_work_weight': 1.1658050258010557, 'successor_release_penalty_weight': 2.9804993527548627, 'wait_decay_power': 0.6090106410575694, 'energy_normalization_scale': 1.1027650185697087, 'slack_feasibility_exponent': 1.1363400283951441, 'ddl_risk_sensitivity': 1.3805884439325284}, 'optimizer_config_hash': '087d89d0b2174a4ef39ab75b5292a4f2a6641d337c4dd6a2bb86ea7b8b713284', 'parameter_diagnostics_hash': 'd5affdbc019cc3795cd06b0c2c7ef834706e5ea190a3e4a844ef1267d3198b80', 'optimizer_seed': 0, 'training_seeds': [0, 1, 2], 'validation_seeds': [3, 4]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with unified DDL-risk gating and zero-branch structure:
      - Replaces conditional `np.where` on `(slack <= median_slack) & (norm_uncertainty > ...)` with a single-element logical product.
      - Uses `np.clip` and `np.power` only — no branching beyond the *required* one DDL gate (now implemented as vectorized arithmetic).
      - All tunables declared and used; no numeric literals beyond {-2,-1,0,1,2}.
      - Enforces finiteness via np.nan_to_num with machine limits.
      - Removes all secondary conditionals (e.g., no `if N > 0` branches — uses `np.where` with safe fallbacks).
    """
    eps = 2.1134145571677787e-06
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    neg_slack = np.clip(-slack, 0.0, None)
    median_slack = np.where(N > 0, np.median(slack), 0.0)
    median_uncertainty = np.where(N > 0, np.median(uncertainty), 0.0)
    ptp_uncertainty = np.where(N > 1, np.ptp(uncertainty), eps)
    norm_uncertainty = (uncertainty - median_uncertainty) / (ptp_uncertainty + eps)
    slack_tight = (slack <= median_slack + eps).astype(float)
    unc_risky = (norm_uncertainty > 0.1882548906640742).astype(float)
    ddl_risk_gate = slack_tight * unc_risky
    critical_path_pressure = upward_rank * remaining_work
    gated_critical_pressure = ddl_risk_gate * critical_path_pressure
    slack_feasibility = np.clip(slack / (np.abs(median_slack) + eps), 0.0, 1.0)
    slack_feasibility_sharpened = np.power(slack_feasibility + eps, 1.1363400283951441)
    successor_release_penalty = remaining_work * (1.0 - slack_feasibility_sharpened)
    gated_successor_penalty = ddl_risk_gate * successor_release_penalty * 2.9804993527548627
    amplified_neg_slack = neg_slack * (1.0 + 1.3805884439325284 * ddl_risk_gate)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / (duration + eps)
    energy_center = np.where(N > 0, np.median(energy_per_duration), 0.0)
    energy_std = np.where(N > 1, np.std(energy_per_duration), 0.0)
    energy_ptp = np.where(N > 0, np.ptp(energy_per_duration), eps)
    energy_dispersion = np.where(energy_std > eps, energy_std, energy_ptp)
    norm_energy = (energy_per_duration - energy_center) / (energy_dispersion * 1.1027650185697087 + eps)
    max_wait = np.where(N > 0, np.max(ready_wait_time), eps)
    wait_ratio = np.clip(ready_wait_time / (max_wait + eps), 0.0, 1.0)
    wait_decay = np.power(wait_ratio + eps, 0.6090106410575694)
    score = amplified_neg_slack + gated_successor_penalty + 1.1658050258010557 * gated_critical_pressure + norm_energy - wait_decay
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
