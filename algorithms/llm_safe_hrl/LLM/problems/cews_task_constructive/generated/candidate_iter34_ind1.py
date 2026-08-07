import numpy as np
RULE_METADATA = {'structure_hash': '892ee381d45d1908c4981089badc73578a881bc387f0170e378d504f5f9c6ce1', 'parameter_schema_hash': 'c0b5f3e02f7fadd42854e475bf680c8305d8da548bf8cc1f1ed8507d6e0ad926', 'best_parameter_hash': 'f22fa35a63c7d0a9483d991631f0f52c776e923905f408f0a111f1a3ee20ce15', 'best_parameters': {'epsilon': 1.4891244358703923e-05, 'ddl_risk_gate_threshold': 0.05106418970470323, 'ddl_risk_gate_slack_ratio': 0.7804614070236375, 'upward_rank_remaining_work_weight': 1.3742791672263477, 'energy_uncertainty_coupling': 0.640047669220691, 'wait_saturation_exponent': 0.9866741664072629}, 'optimizer_config_hash': '087d89d0b2174a4ef39ab75b5292a4f2a6641d337c4dd6a2bb86ea7b8b713284', 'parameter_diagnostics_hash': '4cc948fd7c8e141efd3dd6f2837537d3b7b1ad45aa42462cdbd792c748bfcc59', 'optimizer_seed': 0, 'training_seeds': [0, 1, 2], 'validation_seeds': [3, 4]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule with structural improvements guided by counterfactual evidence:
      - Introduces joint DDL-risk protection gate: activates bottleneck & successor-release terms only when
        (slack <= median_slack * ddl_risk_gate_slack_ratio) AND (normalized_uncertainty > ddl_risk_gate_threshold)
      - Replaces sigmoid wait saturation with bounded power-law saturation for smoother, more robust anti-starvation.
      - Uses raw upward_rank × remaining_work as uncoupled critical-path pressure (no multiplicative urgency coupling).
      - Adds uncertainty-modulated energy penalty: higher uncertainty → stronger marginal energy penalty.
      - Removes all adaptive normalization on neg_slack and urgency_linear to preserve hard-deadline dominance.
      - All operations are finite, deterministic, and use only {-2,-1,0,1,2} literals.
    """
    eps = 1.4891244358703923e-05
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    neg_slack = np.clip(-slack, 0.0, None)
    median_slack = np.median(slack) if N > 0 else 0.0
    high_risk_slack_condition = slack <= median_slack * 0.7804614070236375 + eps
    unc_min = np.min(uncertainty) if N > 0 else 0.0
    unc_max = np.max(uncertainty) if N > 0 else eps
    normalized_uncertainty = (uncertainty - unc_min) / (unc_max - unc_min + eps)
    high_uncertainty_condition = normalized_uncertainty > 0.05106418970470323
    ddl_risk_gate = np.where(high_risk_slack_condition & high_uncertainty_condition, 1.0, 0.0)
    critical_path_pressure = upward_rank * remaining_work
    energy_penalty = min_incremental_energy * (1.0 + 0.640047669220691 * uncertainty)
    wait_scaled = np.clip(ready_wait_time / (np.maximum(np.mean(ready_wait_time), eps) + eps), 0.0, 2.0)
    wait_saturation = np.power(wait_scaled + eps, 0.9866741664072629)
    score = neg_slack + ddl_risk_gate * 1.3742791672263477 * critical_path_pressure + energy_penalty - wait_saturation
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
