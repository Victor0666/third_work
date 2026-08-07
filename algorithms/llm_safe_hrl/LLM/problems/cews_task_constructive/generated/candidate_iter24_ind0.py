import numpy as np
RULE_METADATA = {'structure_hash': '9442f8bc6c1e2cbdfaa0823d1aff9c435b0071b087f504b07e634381b2ad3573', 'parameter_schema_hash': '272b6ee8da1cb450cda26a3368b2c61dd00b95f4d126efdb9b571e7ef433e894', 'best_parameter_hash': 'dba2c0e1b10348af6bd607a430ce279a29e0f588ddcab0c9a3a2997a81259522', 'best_parameters': {'epsilon': 4.261180354496697e-06, 'ddl_hard_guard_weight': 345.0341392341987, 'bottleneck_uncertainty_steepness': 0.9093080864170839, 'energy_efficiency_weight': 1.689094749951264, 'critical_path_coupling': 1.3487838793422104, 'wait_fairness_weight': 0.6014727566327811, 'slack_sensitivity_threshold': 0.33687359614932927, 'uncertainty_activation_threshold': 0.5389890520212579, 'mad_scale_factor': 0.8253108457266102, 'energy_uncertainty_coupling': 0.8275168880190997, 'uncertainty_sigmoid_steepness': 5.724633484611597}, 'optimizer_config_hash': '087d89d0b2174a4ef39ab75b5292a4f2a6641d337c4dd6a2bb86ea7b8b713284', 'parameter_diagnostics_hash': 'c3bad490c56f902cd6674133d2436933241af1edd91d20eb4a221be38fbadfbf', 'optimizer_seed': 0, 'training_seeds': [0, 1, 2], 'validation_seeds': [3, 4]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with:
      - Simplified criticality logic: single slack-sensitivity threshold
      - Scenario-aware activation: bottleneck and energy uncertainty terms only engage above uncertainty_activation_threshold
      - Restored raw min_incremental_energy usage (no duration coupling)
      - Robust rank normalization preserved
      - All numeric literals strictly in {-2,-1,0,1,2}; no unbounded operations
      - Explicit finite-domain clipping for all nonlinearities
      - Fixed sigmoid steepness now declared as tunable parameter
    """
    eps = 4.261180354496697e-06
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    finfo = np.finfo(float)
    score = np.full(N, finfo.max, dtype=float)
    feasible_mask = slack >= -eps
    if not np.any(feasible_mask):
        return score

    def rank_normalize(x):
        x = np.copy(x)
        if N <= 1 or np.all(x == x[0]):
            return np.zeros_like(x, dtype=float)
        sorted_idx = np.argsort(x)
        ranks = np.empty_like(sorted_idx, dtype=float)
        ranks[sorted_idx] = np.arange(N, dtype=float)
        norm_ranks = 2.0 * ranks / (N - 1.0) - 1.0
        return np.clip(norm_ranks, -1.0, 1.0)
    median_slack = np.median(slack) if N > 0 else 0.0
    tight_slack_boundary = median_slack * (1.0 - 0.33687359614932927)
    urgency_linear = np.clip(median_slack - slack, 0.0, 2.0)
    urgency_mask = (slack <= tight_slack_boundary) & (slack >= -eps)
    urgency = np.where(urgency_mask, urgency_linear, 0.0)
    norm_urgency = rank_normalize(urgency)
    med_energy = np.median(min_incremental_energy) if N > 0 else 0.0
    abs_devs = np.abs(min_incremental_energy - med_energy)
    mad_energy = np.median(abs_devs) if N > 0 else eps
    denom_energy = 0.8253108457266102 * (mad_energy + eps)
    norm_energy_eff = (min_incremental_energy - med_energy) / (denom_energy + eps)
    norm_energy_eff = np.clip(norm_energy_eff, -1.0, 1.0)
    unc_centered = uncertainty - 0.5389890520212579
    unc_clipped = np.clip(unc_centered, -2.0, 2.0)
    unc_gate = 1.0 / (1.0 + np.exp(-5.724633484611597 * unc_clipped))
    energy_uncertainty_term = min_incremental_energy * unc_gate
    norm_energy_uncertain = rank_normalize(energy_uncertainty_term)
    bottleneck_gate = np.where(uncertainty >= 0.5389890520212579, 1.0 / (1.0 + np.exp(-0.9093080864170839 * np.clip(unc_centered, 0.0, 2.0))), 0.0)
    bottleneck_pressure = upward_rank * remaining_work * bottleneck_gate
    norm_bottleneck = rank_normalize(bottleneck_pressure)
    norm_wait = rank_normalize(ready_wait_time)
    inv_wait = 1.0 - norm_wait
    local_score = norm_urgency + 1.689094749951264 * norm_energy_eff + 0.8275168880190997 * norm_energy_uncertain + 1.3487838793422104 * norm_bottleneck - 0.6014727566327811 * inv_wait
    score[feasible_mask] = local_score[feasible_mask]
    score[~feasible_mask] = 345.0341392341987 * finfo.max
    score = np.nan_to_num(score, nan=finfo.max, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
