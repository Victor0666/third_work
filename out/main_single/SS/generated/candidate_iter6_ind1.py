import numpy as np
RULE_METADATA = {'structure_hash': '3ca63ab09a86e272e858f6665e350f3cda3c392bef12b580ce83f61d2f078837', 'parameter_schema_hash': 'd92dd9c34c394af8b80f7d26821c124cf0a21a202669bad2a06551a2d829fce0', 'best_parameter_hash': '8f693254ee653ad6635c8127f5b2a65d0abeca3015d47ce6e75498c322e80949', 'best_parameters': {'epsilon': 0.00010568137857829262, 'slack_penalty_exponent': 1.3932436551069203, 'criticality_scale': 2.509287457721557, 'energy_sensitivity': 0.5250960294743654, 'duration_robustness': 0.1508582358313188, 'wait_decay': 0.25986814337501346, 'uncertainty_gate_threshold': 0.2702911106457979, 'slack_gate_width': 0.8108702764705609, 'energy_slack_interaction': 0.5529862909427605, 'rank_slack_coupling': 0.823594253520878, 'uncertainty_smoothness': 2.2266285711967937, 'negative_slack_only_activation': 0.765874482184776}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': '026697792a01feb5141caa2539e33372be331a8a47ecb716cd7f3a21e2ef1f19', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: removes destabilizing successor_release_score; restores monotonic clipped-power slack pressure; 
       introduces negative-slack-only activation for uncertainty-duration penalty to enforce hard deadline feasibility;
       retains robust median/MAD normalization, smooth uncertainty sigmoid, restored rank-slack coupling, and saturating wait relief."""
    eps = 0.00010568137857829262
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
        if N == 0:
            return np.zeros(0, dtype=float)
        x_abs = np.abs(x)
        center = np.median(x_abs)
        dev = x_abs - center
        scale = np.median(np.abs(dev)) + eps
        return dev / (scale + eps)
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    slack_pressure = np.power(np.clip(-norm_slack, 0.0, None), 1.3932436551069203)
    rank_gate = np.clip(1.0 - norm_slack / (0.8108702764705609 + eps), 0.0, 1.0)
    boosted_rank = norm_rank * (1.0 + 2.509287457721557 * rank_gate)
    uncertainty_activation = 1.0 / (1.0 + np.exp(-2.2266285711967937 * (norm_uncert - 0.2702911106457979)))
    negative_slack_mask = (slack < 0.0).astype(float) * 0.765874482184776
    duration_risk_interaction = norm_duration * uncertainty_activation * 0.1508582358313188 * negative_slack_mask
    wait_benefit = 1.0 - np.exp(-0.25986814337501346 * (norm_wait + eps))
    energy_slack_penalty = norm_energy * (1.0 + 0.5529862909427605 * slack_pressure)
    coupled_rank_reward = norm_rank * (1.0 + 0.823594253520878 * slack_pressure)
    score = +slack_pressure + energy_slack_penalty + 0.5250960294743654 * norm_energy - coupled_rank_reward - wait_benefit + duration_risk_interaction
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
