import numpy as np
RULE_METADATA = {'structure_hash': '60c2989b63759cd3a172bdf9d008e06089e1be57bc35222a848f026d5fd0b755', 'parameter_schema_hash': '994283c7957e1fb1c7e923e47ac6b7d47a6d9651978f5af6b8a52a6663dc6efd', 'best_parameter_hash': '06be3404ee4e3b78bd756c47d7edb3e692b8df87c806c48dab26bc1120cee0a8', 'best_parameters': {'epsilon': 0.0007978960601594918, 'slack_urgency_scale': 3.3073520379213304, 'criticality_scale': 1.0916842432387521, 'energy_sensitivity': 1.9441741050791344, 'wait_benefit_decay': 0.19944795444462193, 'uncertainty_gate_threshold': 0.04091637197940665, 'remaining_work_weight': 1.3434506246400522, 'wait_saturation_offset': 8.700086070288334e-06, 'energy_uncertainty_interaction': 0.4299573964656906, 'slack_tightness_threshold': -0.7180750832160452, 'local_release_bonus_weight': 0.6188034070060854}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': 'fd264c50c52d2dfd98ef74e87ea3e621257ca3cc12df3da3d871fe27ee6cd900', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: replaces dual-gated urgency with *relative-slack* tanh urgency
       (tanh(-slack / (mean|slack| + eps))) for improved gradient sensitivity near deadlines;
       reverts successor-release to additive local-gated term (no coupling); removes global ddl_protection_boost
       to avoid diluting per-task urgency; retains robust mean-abs normalization and bounded tanh anti-starvation.
    """
    eps = 0.0007978960601594918
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    def robust_norm(x):
        x_abs = np.abs(x)
        denom = np.mean(x_abs) + eps
        return x / denom
    norm_slack = robust_norm(slack)
    norm_energy = robust_norm(min_incremental_energy)
    norm_duration = robust_norm(min_exec_time + min_comm_time)
    norm_rank = robust_norm(upward_rank)
    norm_work = robust_norm(remaining_work)
    norm_wait = robust_norm(ready_wait_time)
    norm_uncert = robust_norm(uncertainty)
    slack_abs_mean = np.mean(np.abs(slack)) + eps
    relative_urgency = -slack / slack_abs_mean
    slack_urgency = np.tanh(3.3073520379213304 * relative_urgency)
    norm_urgency = robust_norm(slack_urgency)
    slack_tight = (slack < 0.0) | (norm_slack < -0.7180750832160452)
    uncert_low = norm_uncert < 0.04091637197940665
    local_ddl_gate = np.where(slack_tight & uncert_low, 1.0, 0.0)
    boosted_rank = norm_rank * (1.0 + 1.0916842432387521 * local_ddl_gate)
    release_bonus = 0.6188034070060854 * norm_work * norm_rank * local_ddl_gate
    wait_benefit = np.tanh(0.19944795444462193 * (norm_wait + 8.700086070288334e-06))
    risk_energy_gate = np.where((slack < 0.0) & (norm_uncert > 0.04091637197940665), 1.0, 0.0)
    energy_uncert_penalty = norm_energy * risk_energy_gate * 0.4299573964656906
    score = +norm_urgency - boosted_rank - 1.9441741050791344 * norm_energy - wait_benefit - release_bonus + energy_uncert_penalty + 1.3434506246400522 * norm_work
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    assert score.shape == (N,), f'Expected shape {(N,)}, got {score.shape}'
    return score.reshape(-1)
