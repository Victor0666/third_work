import numpy as np
RULE_METADATA = {'structure_hash': '3ec6ccd9a51976e0e7e8a36bc781b766422fcaf7b5908c2f1fbf302efa1e7abe', 'parameter_schema_hash': 'af2567f35ce913116b6f68738f37cd1ec50cd7366d3182f3ca29d423bd6974a3', 'best_parameter_hash': 'eefb4c7b56d7813d438b6f671772298040574820965d0578e6cf05790bcd6358', 'best_parameters': {'epsilon': 8.636524705006501e-05, 'slack_penalty_exponent': 2.9168031728543795, 'criticality_scale': 0.5956789212405499, 'energy_sensitivity': 0.7137990697748751, 'uncertainty_gate_threshold': 0.04505429652975669, 'slack_pressure_gate_steepness': 1.8913307355444213, 'remaining_work_weight': 0.04067722670066992, 'wait_decay': 0.7666392649537046, 'ddl_protection_gate_slope': 7.143290232685041, 'energy_uncertainty_interaction': 0.6074271095302874, 'starvation_saturation_offset': 0.23882889933177676, 'duration_risk_coupling_strength': 0.0796001748191178}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': 'bb23b386cb4d04c48afc905d6b4b0f016f2dc31f672125c51bb8f7fc24accf55', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule: merges Parent 2's robust hinge & joint gate with Parent 1's synthetic load proxy;
       introduces starvation_saturation_offset for sharper low-wait discrimination;
       replaces redundant duration_robustness with duration_risk_coupling_strength to unify congestion modeling;
       retains validated softplus starvation, clipped linear urgency, and unified joint slack–uncertainty gating;
       eliminates all unbounded ops and ensures strict [-2,2] clipping per term to guarantee stability under CMA-ES."""
    eps = 8.636524705006501e-05
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
    slack_feasibility = 1.0 / (1.0 + np.exp(-7.143290232685041 * slack))
    uncert_feasibility = 1.0 / (1.0 + np.exp(7.143290232685041 * (norm_uncert - 0.04505429652975669)))
    joint_gate = slack_feasibility * uncert_feasibility
    ddl_breach = (slack <= 0.0).astype(float)
    critical_path_leverage = norm_rank * norm_work * ddl_breach
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 2.9168031728543795
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    slack_hinge = np.clip(-norm_slack, 0.0, 1.0)
    rank_slack_coupling = 1.0 + 0.5956789212405499 * slack_hinge
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-1.8913307355444213 * (slack_pressure - 1.0)))
    boosted_rank = norm_rank * rank_slack_coupling * (1.0 + 0.5956789212405499 * rank_gate)
    wait_benefit = np.log1p(np.exp(-0.7666392649537046 * (ready_wait_time + 0.23882889933177676)))
    congestion_proxy = 0.0796001748191178 * norm_duration * norm_uncert * joint_gate
    energy_uncert_penalty = norm_energy * norm_uncert * joint_gate
    energy_slack_penalty = norm_energy * slack_pressure * joint_gate
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(critical_path_leverage, -2.0, 2.0) - np.clip(boosted_rank, -2.0, 2.0) - 0.7137990697748751 * np.clip(norm_energy * joint_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + 0.6074271095302874 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 2.9168031728543795 * np.clip(energy_slack_penalty, -2.0, 2.0) + 0.04067722670066992 * np.clip(norm_work, -2.0, 2.0) + np.clip(congestion_proxy, -2.0, 2.0)
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.reshape(-1)
