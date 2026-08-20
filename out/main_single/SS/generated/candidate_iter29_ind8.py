import numpy as np
RULE_METADATA = {'structure_hash': 'f4151174673b6f36411c2065ec7cf5beeabfcf0523dd6539767882bca1e374eb', 'parameter_schema_hash': '41991dbf425a0eb470550142ac98ba4074c0d7c5f26c9247b747d1089e8a958d', 'best_parameter_hash': '8d85bcc3a19534162fe60990bf0325ea5123fdb36c8063e87d81d1ea2e083edf', 'best_parameters': {'epsilon': 1.0295351265209894e-06, 'slack_penalty_exponent': 3.1807992740232907, 'criticality_scale': 2.049425267985389, 'energy_sensitivity': 1.0248038149253855, 'duration_robustness': 0.31028485554821605, 'wait_decay': 0.10527307454607232, 'uncertainty_gate_threshold': 0.48648813461329726, 'slack_pressure_gate_steepness': 3.128493546822939, 'remaining_work_weight': 0.00975704632173794, 'wait_saturation_offset': 2.1331785710056056e-08, 'energy_uncertainty_interaction': 0.12195281744408532, 'ddl_protection_gate_slope': 7.169852453503912}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': 'e5a091802f02d6fa5108067219a44a62fa899d1756c2e30699f0e4856ac87dde', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's robust DDL-protection and duration_robustness with Parent 1's validated successor-release coupling.
       Structural novelty: replaces scalar critical_path_leverage with gated, coupling-weighted successor_release term activated only on breach and uncertainty,
       and adds a bounded linear slack-rank interaction that preserves monotonic urgency while avoiding over-parameterization.
       All terms clipped to [-2,2]; median-MAD normalization ensures outlier resilience; final score finite and shape-(N,)."""
    eps = 1.0295351265209894e-06
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
        if N == 1:
            med = x[0]
            mad = eps
        else:
            med = np.median(x)
            mad = np.median(np.abs(x - med))
        spread = mad if mad > eps else eps
        return (x - med) / spread
    norm_slack = median_mad_normalize(slack)
    norm_energy = median_mad_normalize(min_incremental_energy)
    norm_duration = median_mad_normalize(min_exec_time + min_comm_time)
    norm_rank = median_mad_normalize(upward_rank)
    norm_work = median_mad_normalize(remaining_work)
    norm_wait = median_mad_normalize(ready_wait_time)
    norm_uncert = median_mad_normalize(uncertainty)
    ddl_gate = 1.0 / (1.0 + np.exp(-7.169852453503912 * slack))
    ddl_breach = (slack <= 0.0).astype(float)
    uncert_gate = 1.0 / (1.0 + np.exp(-7.169852453503912 * (norm_uncert - 0.48648813461329726)))
    successor_release = norm_rank * norm_work * ddl_breach * uncert_gate
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 3.1807992740232907
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    coupled_slack = np.clip(norm_slack, -1.0, 1.0)
    rank_slack_coupling = 1.0 + 2.049425267985389 * coupled_slack
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-3.128493546822939 * (slack_pressure - 1.0)))
    boosted_rank = norm_rank * rank_slack_coupling * (1.0 + 2.049425267985389 * rank_gate)
    duration_risk_score = norm_duration * uncert_gate * slack_pressure * ddl_gate
    wait_benefit = 1.0 - np.exp(-0.10527307454607232 * (ready_wait_time + 2.1331785710056056e-08))
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_gate
    energy_slack_penalty = norm_energy * (1.0 + 3.1807992740232907 * slack_pressure) * ddl_gate
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(successor_release, -2.0, 2.0) - np.clip(boosted_rank, -2.0, 2.0) - 1.0248038149253855 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + 0.31028485554821605 * np.clip(duration_risk_score, -2.0, 2.0) + 0.12195281744408532 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 3.1807992740232907 * np.clip(energy_slack_penalty, -2.0, 2.0) + 0.00975704632173794 * np.clip(norm_work, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
