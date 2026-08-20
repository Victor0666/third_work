import numpy as np
RULE_METADATA = {'structure_hash': 'da8638972ff393ff7d52cbd53b2aa585e3b45ef0d84b93c889f174c9008818e8', 'parameter_schema_hash': 'c36b4ac2328dc6c31a100d01c6474a110c45095bb0168f5beb7aedce1c70781f', 'best_parameter_hash': '720435f7164ed2201d107fa9ab990623bcac109e6104cb4d29ad47a94ade30eb', 'best_parameters': {'epsilon': 7.462505388805668e-05, 'slack_penalty_exponent': 3.383104374044188, 'criticality_scale': 0.7732462481535524, 'energy_sensitivity': 1.2264275009580745, 'duration_robustness': 0.0452532089891526, 'wait_decay': 0.005684230283941586, 'uncertainty_gate_threshold': 0.4545762381917087, 'slack_pressure_gate_steepness': 1.4652099198305049, 'remaining_work_weight': 0.42034517442791675, 'wait_saturation_offset': 0.0003758481139239937, 'energy_uncertainty_interaction': 0.44452072389002667, 'successor_release_gate_threshold': 0.28239326674263776}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': '741964448194856ccdcf313e47de63adca2cb911ecbe90e07cb6e0fe472b4715', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: introduces conditional successor-release coupling gated by slack_pressure only;
       uses norm_duration for scale-consistent risk scoring; enforces hard DDL protection via +inf clamping;
       removes rank quantile gating to comply with 12-parameter limit while preserving structural intent.
    """
    eps = 7.462505388805668e-05
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
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        spread = mad if mad > eps else eps
        return (x - med) / spread
    norm_slack = median_mad_normalize(slack)
    norm_energy = median_mad_normalize(min_incremental_energy)
    norm_rank = median_mad_normalize(upward_rank)
    norm_work = median_mad_normalize(remaining_work)
    norm_wait = median_mad_normalize(ready_wait_time)
    norm_uncert = median_mad_normalize(uncertainty)
    norm_duration = median_mad_normalize(min_exec_time + min_comm_time)
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 3.383104374044188
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-1.4652099198305049 * (slack_pressure - 1.0)))
    successor_release_active = np.where(slack_pressure > 0.28239326674263776, 1.0, 0.0)
    critical_path_leverage = norm_rank * norm_work * successor_release_active
    boosted_rank = norm_rank * (1.0 + 0.7732462481535524 * rank_gate)
    uncert_gate = np.where(norm_uncert > 0.4545762381917087, 1.0, 0.0)
    duration_risk_score = norm_duration * uncert_gate * slack_pressure
    wait_benefit = 1.0 - np.exp(-0.005684230283941586 * (ready_wait_time + 0.0003758481139239937))
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate
    score = +norm_slack_penalty - boosted_rank - 0.7732462481535524 * critical_path_leverage - 1.2264275009580745 * norm_energy - wait_benefit + 0.0452532089891526 * duration_risk_score + 0.44452072389002667 * energy_uncert_penalty + 0.42034517442791675 * norm_work
    deadline_violated = slack < -eps
    score = np.where(deadline_violated, np.finfo(float).max, score)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
