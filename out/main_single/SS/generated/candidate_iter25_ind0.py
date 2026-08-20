import numpy as np
RULE_METADATA = {'structure_hash': 'a91e146eb056bcc4bcd2253027177eab958d8049e9d4b4b82fef6f10f849fc21', 'parameter_schema_hash': 'aee2c549526007dd2b18e76d6f79401532e41b1766033817a4d4cf71dfcd3b2a', 'best_parameter_hash': '00632b46ede9da01623ad12625df9a3fdcd125e7baaa86dea2226a5001e0d999', 'best_parameters': {'epsilon': 2.2358622292374672e-05, 'slack_penalty_exponent': 1.8732100644132201, 'criticality_scale': 1.694197328362718, 'energy_sensitivity': 0.14803162072630477, 'energy_uncertainty_interaction': 0.006350404615985947, 'remaining_work_weight': 1.1768416059258766, 'uncertainty_gate_threshold': 0.35742826832656127, 'wait_decay': 0.5380747451121541}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': '7814ba9d914eec563013e6548e02167b0f70b24b93cad7b1c4b429fa84de47e7', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule: removes unused parameters; retains successor-release interaction;
       uses joint congestion gating (high ready_wait_time AND high uncertainty);
       applies energy/uncertainty penalties only under feasibility (slack >= 0);
       replaces sigmoid gates with direct threshold-based logic for stability and evidence alignment."""
    eps = 2.2358622292374672e-05
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def robust_mad_normalize(x):
        x = np.copy(x)
        if N == 1:
            med = x[0]
            mad = eps
        else:
            med = np.median(x)
            mad = np.median(np.abs(x - med))
        spread = np.maximum(mad, eps)
        return (x - med) / spread
    norm_slack = robust_mad_normalize(slack)
    norm_energy = robust_mad_normalize(min_incremental_energy)
    norm_rank = robust_mad_normalize(upward_rank)
    norm_work = robust_mad_normalize(remaining_work)
    norm_wait = robust_mad_normalize(ready_wait_time)
    norm_uncert = robust_mad_normalize(uncertainty)
    feasible_mask = (slack >= 0.0).astype(float)
    ddl_breach = (slack <= 0.0).astype(float)
    critical_path_leverage = norm_rank * norm_work * ddl_breach
    successor_release = np.clip(min_exec_time, eps, None) * np.clip(norm_rank, -1.0, 1.0) * ddl_breach
    slack_mad = np.maximum(np.median(np.abs(slack - np.median(slack))), eps)
    norm_slack_urgency = (slack - np.median(slack)) / (slack_mad + eps)
    raw_slack_penalty = np.where(slack < 0.0, (-slack) ** 1.8732100644132201, 0.0)
    norm_slack_penalty = robust_mad_normalize(raw_slack_penalty)
    wait_centered = ready_wait_time - np.median(ready_wait_time)
    wait_std = np.maximum(np.std(ready_wait_time), eps)
    wait_z = wait_centered / (wait_std + eps)
    uncert_centered = uncertainty - np.median(uncertainty)
    uncert_std = np.maximum(np.std(uncertainty), eps)
    uncert_z = uncert_centered / (uncert_std + eps)
    congestion_gate = (wait_z > 0.35742826832656127) & (uncert_z > 0.35742826832656127)
    congestion_gate = congestion_gate.astype(float)
    wait_benefit = np.exp(-0.5380747451121541 * ready_wait_time) * (1.0 - feasible_mask)
    energy_uncert_penalty = norm_energy * norm_uncert * feasible_mask * congestion_gate
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(critical_path_leverage, -2.0, 2.0) - np.clip(successor_release, -2.0, 2.0) - np.clip(norm_slack_urgency, -2.0, 2.0) - 0.14803162072630477 * np.clip(norm_energy * feasible_mask, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + 0.006350404615985947 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 1.1768416059258766 * np.clip(norm_work, -2.0, 2.0) + 1.694197328362718 * np.clip(norm_rank * feasible_mask, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
