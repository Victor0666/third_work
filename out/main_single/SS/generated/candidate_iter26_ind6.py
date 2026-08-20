import numpy as np
RULE_METADATA = {'structure_hash': 'a6a4a64aab5a5a68e03df011b87ba1c826d8087d472aa9400cb06b6844e8fdad', 'parameter_schema_hash': 'edfdbc985b8b26d3774c9a08b2118fe15932dc4d4994fe6842a3a57fc5ecb594', 'best_parameter_hash': '5240d9654520bb35a528e8d632deb2a4087eabc2b7f43c2b5261dd68cfb71daf', 'best_parameters': {'epsilon': 0.003751929728686598, 'slack_penalty_exponent': 1.048710172065259, 'criticality_scale': 0.7235530100321521, 'energy_sensitivity': 0.12032564449686012, 'duration_robustness': 1.7683140013078873, 'wait_decay': 0.655756466362098, 'uncertainty_gate_threshold': 0.0003895293224667693, 'slack_pressure_gate_steepness': 6.1419644977195675, 'remaining_work_weight': 1.0699146608111325, 'wait_saturation_offset': 1.6188828490046432e-05, 'energy_uncertainty_interaction': 0.28152560307713614, 'ddl_protection_gate_slope': 9.850911703458344}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': '70edc59b31bfd28885232be6b21981828172ddc86687c5fe1349f99dc45a68e1', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's robust median-MAD normalization and sharpened slack gates 
       with Parent 1's validated critical-path coupling and slack-aware uncertainty gating.
       Key improvements:
         - Replaces fixed uncertainty_gate_threshold with adaptive version: threshold rises as slack approaches zero
         - Introduces dedicated starvation relief *only* under deadline pressure (slack <= 0), decoupled from congestion
         - Combines both critical-path signals: (1) boosted_rank (Parent 2) + (2) critical_coupling_under_pressure (Parent 1)
         - Uses clipped MAD-normalized slack for both penalty and gating to enhance sensitivity near deadlines
         - All terms bounded via [-2,2] clipping; final score deterministic, finite, shape-(N,)."""
    eps = 0.003751929728686598
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
    adaptive_uncert_thresh = 0.0003895293224667693 * (1.0 - 1.0 / (1.0 + np.exp(-2.0 * slack)))
    uncert_gate_adaptive = (norm_uncert > adaptive_uncert_thresh).astype(float) * (slack >= 0.0).astype(float)
    ddl_gate = 1.0 / (1.0 + np.exp(-9.850911703458344 * slack))
    ddl_breach = (slack <= 0.0).astype(float)
    critical_coupling = norm_rank * norm_work * ddl_breach * 0.7235530100321521
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.048710172065259
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    coupled_slack = np.clip(norm_slack, -1.0, 1.0)
    rank_slack_coupling = 1.0 + 0.7235530100321521 * coupled_slack
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-6.1419644977195675 * (slack_pressure - 1.0)))
    boosted_rank = norm_rank * rank_slack_coupling * (1.0 + 0.7235530100321521 * rank_gate)
    duration_risk_score = norm_duration * uncert_gate_adaptive * slack_pressure * ddl_gate
    wait_benefit_pressure = (1.0 - np.exp(-0.655756466362098 * (ready_wait_time + 1.6188828490046432e-05))) * ddl_breach
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate_adaptive * ddl_gate
    energy_slack_penalty = norm_energy * (1.0 + 1.048710172065259 * slack_pressure) * ddl_gate
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(critical_coupling, -2.0, 2.0) - np.clip(boosted_rank, -2.0, 2.0) - 0.12032564449686012 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - np.clip(wait_benefit_pressure, -2.0, 2.0) + 1.7683140013078873 * np.clip(duration_risk_score, -2.0, 2.0) + 0.28152560307713614 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 1.048710172065259 * np.clip(energy_slack_penalty, -2.0, 2.0) + 1.0699146608111325 * np.clip(norm_work, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
