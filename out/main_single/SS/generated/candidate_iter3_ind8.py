import numpy as np
RULE_METADATA = {'structure_hash': '4da3a434b069c3a0e44621d115191dc3be0b601b4d0211b009341b93dfeae362', 'parameter_schema_hash': '5266c2efddb2a9fa835716bb42913880779e619d6bf4087a0fcda69eb7eb0ab1', 'best_parameter_hash': '2197d20ef792d028db76c57fbfd883782ba2349700851db02de4b0f0548c8720', 'best_parameters': {'epsilon': 3.149803210218887e-05, 'slack_penalty_exponent': 1.708364072661739, 'criticality_scale': 2.0782768324284158, 'energy_sensitivity': 1.454865077261015, 'duration_robustness': 1.2434827835811553, 'wait_decay': 0.8365985753982863, 'uncertainty_gate_threshold': 0.9759475428617183, 'slack_gate_width': 1.8040304445616961, 'rank_slack_coupling': 0.8563726520209674}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': '93a7b3d383c0e28b2b354db43203fcff5ed09f1bc1445febf2bb8069b69c776d', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: replaces MAD normalization with bounded min-max scaling
       to preserve deadline-critical slack/uncertainty signal fidelity; reintroduces
       energy_sensitivity as core linear term; removes redundant work/energy coupling;
       uses raw slack for pressure computation (not normalized) to avoid distortion
       under hard DDL constraints; all operations protected against NaN/inf."""
    eps = 3.149803210218887e-05
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def bounded_minmax(x):
        x_min = np.min(x)
        x_max = np.max(x)
        range_val = x_max - x_min + eps
        return (x - x_min) / range_val
    norm_energy = bounded_minmax(min_incremental_energy)
    norm_duration = bounded_minmax(min_exec_time + min_comm_time)
    norm_rank = bounded_minmax(upward_rank)
    norm_wait = bounded_minmax(ready_wait_time)
    norm_uncert = bounded_minmax(uncertainty)
    slack_pressure_raw = np.clip(-slack, 0.0, None)
    slack_pressure = np.power(slack_pressure_raw + eps, 1.708364072661739)
    norm_slack_for_gate = bounded_minmax(slack)
    rank_gate = np.clip(1.0 - 2.0 * np.clip(norm_slack_for_gate, 0.0, 1.8040304445616961), 0.0, 1.0)
    boosted_rank = norm_rank * (1.0 + 2.0782768324284158 * rank_gate)
    joint_risk_gate = ((norm_uncert > 0.9759475428617183) & (slack_pressure > 0)).astype(float)
    duration_risk_interaction = norm_duration * joint_risk_gate * 1.2434827835811553
    wait_benefit = np.tanh(0.8365985753982863 * norm_wait)
    coupled_rank = np.clip(norm_rank * (1.0 + 0.8563726520209674 * slack_pressure), 0.0, 2.0)
    score = +slack_pressure + 1.454865077261015 * norm_energy - coupled_rank - wait_benefit + duration_risk_interaction
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    return score
