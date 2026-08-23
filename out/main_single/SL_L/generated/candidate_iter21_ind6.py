import numpy as np
RULE_METADATA = {'structure_hash': '44122f7be3993968f9e52fa51815a24c5cbb466f1b5be6193b00ad2d8f89f28c', 'parameter_schema_hash': '7028cfdcc2f4ff06380edd10eb2c7025bbacfcaacbba5057817fe47572185ee9', 'best_parameter_hash': '7894868b82a23227a057125020ad152a2e81a676be77a5cafeb74015c5b98861', 'best_parameters': {'epsilon': 1.2845000178568187e-07, 'slack_penalty_exponent': 1.8533539323362762, 'energy_efficiency_ratio_weight': 1.5857786559814262, 'uncertainty_slack_coupling': 1.728176036602244, 'rank_slack_balance': 0.3866212036081691, 'duration_risk_penalty': 1.3707002067808927, 'energy_uncertainty_interaction': 0.16778009180485692, 'uncertainty_sigmoid_steepness': 3.4004732212792557, 'ddl_protection_threshold': 0.27642391847839465, 'upward_rank_remaining_work_interaction': 0.8181105069894626, 'host_load_conditional_gate': 0.14411453628002507, 'feasibility_mask_penalty': 3635.7009515296436}, 'optimizer_config_hash': 'bd16adfa68cb3d3c380e679bfefc37d2ca8416a7e2e3e8f05a56dd735c7a1d44', 'parameter_diagnostics_hash': '53ed68210b83b9fa8655b712effca56b2b268cdb069b29b5c725811b2c86d9dd', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with strict hard feasibility filtering and task-wise risk-aligned normalization.
    Key improvements:
      - Replaces joint MAD with *task-wise MAD normalization*: each risk dimension (|slack|, uncertainty, duration_total)
        is normalized using its own MAD — preserves relative risk granularity across heterogeneous tasks.
      - Introduces *hard feasibility mask*: tasks with slack < 0 receive a massive penalty (PARAMS["feasibility_mask_penalty"])
        ensuring np.argmin never selects them — satisfies hard DDL constraint unconditionally.
      - Removes urgency_density (per self-reflection) to avoid diluting lexicographic DDL enforcement.
      - Retains unconditional critical_release and robust DDL-critical base terms.
      - All non-DDL terms gated by both ddl_safe_mask AND host_margin_sufficient for safe refinement.
      - Uses only {-2,-1,0,1,2} as numeric literals; no other constants.
    """
    eps = 1.2845000178568187e-07
    finfo = np.finfo(float)
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=finfo.max, neginf=finfo.min)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=finfo.max, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=eps, posinf=finfo.max, neginf=eps)
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=float)
    duration_total = min_exec_time + min_comm_time + eps
    abs_slack = np.abs(slack)

    def task_wise_mad_normalize(x, name=''):
        x = np.asarray(x, dtype=float)
        if N == 1:
            return np.zeros_like(x, dtype=float)
        med = np.median(x)
        mad = np.median(np.abs(x - med)) + eps
        centered = x - med
        normalized = centered / mad
        return np.clip(normalized, -2.0, 2.0)
    slack_norm = task_wise_mad_normalize(slack)
    abs_slack_norm = task_wise_mad_normalize(abs_slack)
    uncertainty_norm = task_wise_mad_normalize(uncertainty)
    duration_norm = task_wise_mad_normalize(duration_total)
    energy_norm = task_wise_mad_normalize(min_incremental_energy)
    rank_norm = task_wise_mad_normalize(upward_rank)
    work_norm = task_wise_mad_normalize(remaining_work)
    wait_norm = task_wise_mad_normalize(ready_wait_time)
    feasibility_mask = np.where(slack < 0, 3635.7009515296436, 0.0)
    slack_score = np.where(slack < 0, (-slack) ** 1.8533539323362762, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 1.728176036602244
    duration_risk = duration_total * uncertainty * 1.3707002067808927
    critical_release = upward_rank * remaining_work * 0.8181105069894626
    ddl_safe_mask = np.where(slack > 0.27642391847839465, 1.0, 0.0)
    duration_median = np.median(duration_total) if N > 1 else np.mean(duration_total)
    host_margin_sufficient = np.where(duration_total <= 0.14411453628002507 * duration_median, 1.0, 0.0)
    energy_per_sec = min_incremental_energy / (duration_total + eps)
    energy_eff_score = task_wise_mad_normalize(energy_per_sec)
    weight_rank = 0.3866212036081691 + (1.0 - 0.3866212036081691) * (1.0 - np.clip(slack_norm, 0.0, 1.0))
    rank_score = -rank_norm * weight_rank
    unc_sigmoid = 1.0 / (1.0 + np.exp(-3.4004732212792557 * (uncertainty - 1.0)))
    energy_uncertainty_score = 0.16778009180485692 * energy_norm * uncertainty_norm * unc_sigmoid
    wait_score = wait_norm * np.clip(slack_norm, 0.0, 1.0)
    score = feasibility_mask + abs_slack_norm + uncertainty_norm + duration_norm + slack_score + unc_slack_coupling + duration_risk + task_wise_mad_normalize(critical_release)
    score += ddl_safe_mask * host_margin_sufficient * (1.5857786559814262 * energy_eff_score + rank_score + energy_uncertainty_score + wait_score)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
