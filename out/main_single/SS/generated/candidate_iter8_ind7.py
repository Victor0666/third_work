import numpy as np
RULE_METADATA = {'structure_hash': 'b40e96ebc89fcd4a94017699760b58252aae9bff83ec3531d4a43466ddefc5e8', 'parameter_schema_hash': '59a7813487fce8ceb3105e7e8ddbe7ba85784e846f71e00b0f2fadae21fe39f6', 'best_parameter_hash': 'a677fa2c960c239583e7a9233e35fb1fad3345059f170294b79a568ed8c72a20', 'best_parameters': {'epsilon': 0.00025412224734302314, 'slack_penalty_exponent': 2.0659085938711907, 'criticality_scale': 1.664199632287399, 'energy_sensitivity': 1.9311845055438885, 'duration_robustness': 0.033927671009703016, 'wait_decay': 0.4247946822065884, 'uncertainty_gate_threshold': 0.20899403997153726, 'slack_pressure_gate_steepness': 1.9058171227446534, 'remaining_work_weight': 1.3657314672186784, 'wait_saturation_offset': 0.00029821744494772367, 'energy_uncertainty_interaction': 0.585295501241209, 'latency_risk_coupling': 0.2827047464272673}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': 'b3c70588bf21f759ffbbc3dacd1ba9de1f325967e980779dd5ee13c66c20b878', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule: combines Parent 2's robust slack penalty normalization with Parent 1's explicit starvation relief and adds novel latency-risk coupling.
       Key improvements:
         - Introduces `latency_risk_coupling`: multiplies (exec+comm) * uncertainty to directly model latency risk exposure
         - Uses raw-slack penalty *before* normalization (Parent 1 insight) but applies MAD normalization *only* to the penalty itself (Parent 2 refinement)
         - Retains sharp sigmoid slack-pressure gating on norm_slack for stable critical-path activation
         - Preserves logistic starvation relief with saturation offset for fairness
         - All 12 parameters used; no numeric literals except -2,-1,0,1,2."""
    eps = 0.00025412224734302314
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
        x_abs = np.abs(x)
        center = np.median(x_abs) if N > 1 else x_abs[0]
        spread = np.median(np.abs(x_abs - center)) if N > 1 else np.abs(x_abs[0] - center) + eps
        return (x_abs - center) / (spread + eps)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    norm_slack = robust_normalize(slack)
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 2.0659085938711907
    norm_slack_penalty = robust_normalize(raw_slack_penalty)
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-1.9058171227446534 * (slack_pressure - 1.0)))
    boosted_rank = norm_rank * (1.0 + 1.664199632287399 * rank_gate)
    latency_risk_score = (min_exec_time + min_comm_time) * uncertainty
    norm_latency_risk = robust_normalize(latency_risk_score)
    uncert_gate = np.where(norm_uncert > 0.20899403997153726, 1.0, 0.0)
    duration_risk_score = norm_duration * uncert_gate * slack_pressure
    wait_benefit = 1.0 - np.exp(-0.4247946822065884 * (norm_wait + 0.00029821744494772367))
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate
    score = +norm_slack_penalty - boosted_rank - 1.9311845055438885 * norm_energy - norm_duration - wait_benefit + 0.033927671009703016 * duration_risk_score + 0.585295501241209 * energy_uncert_penalty + 1.3657314672186784 * norm_work + 0.2827047464272673 * norm_latency_risk
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
