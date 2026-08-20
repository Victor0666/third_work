import numpy as np
RULE_METADATA = {'structure_hash': '1affdbb51bc5d014d8a68adad76563c447814d8e7797e27c65c7f0457e42a811', 'parameter_schema_hash': 'b871fb5a79e81c33b18732126fa0b774d73808fd3f5836a07cd4ba3f8cb84d93', 'best_parameter_hash': '2aca7e6fe7d0954708bf1fd11d6946ec9d2d0a809a79ee2403e2900d2f14801c', 'best_parameters': {'epsilon': 0.0219064528874319, 'slack_penalty_exponent': 2.8952467565108555, 'criticality_scale': 0.5948394054003986, 'energy_sensitivity': 0.7332043855119993, 'duration_robustness': 0.15510990305958816, 'wait_decay': 0.6386395157572966, 'uncertainty_gate_threshold': 0.614588908476617, 'slack_gate_width': 1.3847719519390438, 'energy_slack_interaction': 1.1808247765169735, 'rank_slack_coupling': 0.6104958295975154, 'uncertainty_smoothness': 2.6983771709061024, 'successor_pressure_weight': 0.9310003254206034}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': 'bacb882d8abd48bf65dcdb9713033d5954ba51fdfb604bf635738ec70dc28366', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining Parent 2's robustness and Parent 1's successor pressure.
    Key novelties:
      - Restored successor-slack pressure term (from Parent 1) using remaining_work as depth-aware urgency proxy
      - Hybrid robust normalization: median-centered MAD + outlier clipping [-2,2] for rank stability
      - Slack pressure drives all key interactions (energy, rank, duration, successor)
      - Smooth uncertainty activation preserves gradient continuity while enabling partial risk response
      - All numeric literals are strictly from {-2,-1,0,1,2}
    """
    eps = 0.0219064528874319
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def robust_normalize_clip(x):
        x = np.asarray(x)
        center = np.median(x)
        mad = np.median(np.abs(x - center))
        scale = mad * 1.0 + eps
        z = (x - center) / scale
        return np.clip(z, -2.0, 2.0)
    norm_slack = robust_normalize_clip(slack)
    norm_energy = robust_normalize_clip(min_incremental_energy)
    norm_duration = robust_normalize_clip(min_exec_time + min_comm_time)
    norm_rank = robust_normalize_clip(upward_rank)
    norm_work = robust_normalize_clip(remaining_work)
    norm_wait = robust_normalize_clip(ready_wait_time)
    norm_uncert = robust_normalize_clip(uncertainty)
    slack_pressure_raw = np.clip(-norm_slack, 0.0, None)
    slack_pressure = np.power(slack_pressure_raw + eps, 2.8952467565108555)
    rank_gate = np.clip(1.0 - norm_slack / (1.3847719519390438 + eps), 0.0, 1.0)
    boosted_rank = norm_rank * (1.0 + 0.5948394054003986 * rank_gate)
    uncertainty_activation = 1.0 / (1.0 + np.exp(-2.6983771709061024 * (norm_uncert - 0.614588908476617)))
    duration_risk_interaction = norm_duration * uncertainty_activation * 0.15510990305958816
    wait_benefit = 1.0 - np.exp(-0.6386395157572966 * (norm_wait + eps))
    energy_slack_penalty = norm_energy * (1.0 + 1.1808247765169735 * slack_pressure_raw)
    coupled_rank_reward = norm_rank * (1.0 + 0.6104958295975154 * slack_pressure_raw)
    successor_pressure = slack_pressure_raw * norm_work
    successor_pressure_term = 0.9310003254206034 * successor_pressure
    score = +slack_pressure + energy_slack_penalty + 0.7332043855119993 * norm_energy - coupled_rank_reward - wait_benefit + duration_risk_interaction + successor_pressure_term
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=finfo.max, posinf=finfo.max, neginf=finfo.min)
    assert score.shape == (N,), f'Expected shape {(N,)}, got {score.shape}'
    return score
