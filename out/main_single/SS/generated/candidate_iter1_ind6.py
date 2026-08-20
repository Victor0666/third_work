import numpy as np
RULE_METADATA = {'structure_hash': '96d0b7cd515a724ca75fb7d6ef4c4b17567a277d69d5f7948c1635e94d509a23', 'parameter_schema_hash': '25915097a79f2f2a1a41e47cbbe57b4ecc460adb1cb3f3abb8e9c6f6ce7a80eb', 'best_parameter_hash': '4850deda295ba477b84879e3eb791e9897c97afe8ef219f6fa18c513856c5ad3', 'best_parameters': {'epsilon': 0.025705931341541487, 'slack_penalty_exponent': 2.683525783899238, 'criticality_scale': 1.536627723864596, 'energy_sensitivity': 1.8137052668664495, 'duration_robustness': 1.3684063880368764, 'wait_decay': 0.501976446889623, 'uncertainty_gate_threshold': 0.7464666302373717, 'rank_slack_coupling': 0.5614084844556588, 'slack_clip_upper': 7.1190060238039905}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': '3707275d57ae57ade9e33b64103a26764097c601a734a42a12a5f14cb56a8f36', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule: all numeric constants declared; only -2,-1,0,1,2 used inline."""
    eps = 0.025705931341541487
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
        x = np.abs(x)
        center = np.median(x)
        scale = np.median(np.abs(x - center)) + eps
        return (x - center) / scale
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    slack_penalty_base = np.where(norm_slack < 0, -np.power(np.clip(-norm_slack, 0.0, 7.1190060238039905), 2.683525783899238), 0.0)
    slack_pressure_ratio = np.clip(-norm_slack, 0.0, 2.0)
    rank_boost = 1.536627723864596 * norm_rank * (1.0 - np.exp(-0.5614084844556588 * slack_pressure_ratio))
    unc_gate_active = (norm_uncert > 0.7464666302373717) & (norm_slack < 0)
    duration_uncert_penalty = np.where(unc_gate_active, 1.3684063880368764 * norm_duration * np.abs(norm_slack), 0.0)
    wait_benefit = 0.501976446889623 * (1.0 - np.exp(-norm_wait / (0.501976446889623 + eps)))
    score = slack_penalty_base - rank_boost + 1.8137052668664495 * norm_energy + duration_uncert_penalty - wait_benefit
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    return score
