import numpy as np
RULE_METADATA = {'structure_hash': 'f9af16bcb8d9f1386def40b4c3bc40ebf808217f1f232e71007fe4793374c9f1', 'parameter_schema_hash': '3db2c7141ae1ac342d05e6153ba6ea9fe1389a8a2f511bffaf7b1c97bae4246c', 'best_parameter_hash': 'c32e047979a31d84f61a3e146fe9743db230cdb8349a178ce0501544971b70e2', 'best_parameters': {'epsilon': 0.08458574286761417, 'slack_risk_penalty': 5.658440857215918, 'slack_urgency_scale': 2.414919855872241, 'energy_efficiency_bias': 2.2736775498780384, 'critical_path_leverage': 1.5347641828430494, 'wait_fairness_gain': 0.02436091936065679, 'uncertainty_sensitivity': 0.40987791682209535, 'duration_balance': 0.6613912373005354, 'slack_pressure_tanh_scale': 0.33598571528207627, 'robustness_mad_factor': 1.5622894931838922, 'successor_release_threshold': 0.2944094389197889}, 'optimizer_config_hash': 'bd16adfa68cb3d3c380e679bfefc37d2ca8416a7e2e3e8f05a56dd735c7a1d44', 'parameter_diagnostics_hash': '6366adfb8c8536a0fd78035e310c951ca6da97eb17a8027f106828f85b71abcf', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining best structural elements from both parents.
    
    Key improvements:
    - Unified slack modeling: linear risk penalty for slack < 0, smooth tanh urgency for slack >= 0
      (no discontinuity at zero; better DDL sensitivity than Parent 1's centered tanh)
    - Successor-release interaction now uses adaptive slack gating with tunable offset
      (replaces rigid 'slack >= 0' with PARAMS['successor_release_threshold'] for CMA-ES tuning)
    - Energy suppression gate strengthened: requires BOTH slack > 0 AND uncertainty <= median_uncertainty
      (prevents premature optimization under risk, per Parent 2 insight)
    - Fairness uses bounded saturating exponential (1 - exp(-gain * norm_wait)) — robust and monotonic
    - All features use MAD normalization with Gaussian-consistent scaling (robust to outliers)
    - Uncertainty amplification applied only to negative slack (tighter causal grounding)
    - Removed fragile work_density_weight and redundant wait_headroom capping (simplifies AST)
    """
    eps = 0.08458574286761417
    finfo = np.finfo(float)
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=finfo.max, neginf=finfo.min)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=finfo.max, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=eps, posinf=finfo.max, neginf=eps)

    def mad_normalize(x):
        x = np.asarray(x, dtype=float)
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        scale = 1.5622894931838922 * (mad if mad > eps else eps)
        return (x - med) / scale
    slack_risk = 5.658440857215918 * np.clip(-slack, 0.0, np.inf)
    slack_urgency = np.tanh(0.33598571528207627 * np.maximum(0.0, slack))
    slack_norm = slack_risk + 2.414919855872241 * (1.0 - slack_urgency)
    duration = min_exec_time + min_comm_time
    duration_norm = mad_normalize(duration)
    energy_norm = mad_normalize(min_incremental_energy)
    median_unc = np.median(uncertainty)
    energy_suppression_gate = ((slack > 0.0) & (uncertainty <= median_unc + eps)).astype(float)
    energy_weight_adj = 2.2736775498780384 * (1.0 - energy_suppression_gate)
    rank_work_interaction = upward_rank * remaining_work
    rank_work_norm = mad_normalize(rank_work_interaction)
    successor_gate = (slack >= 0.2944094389197889).astype(float)
    successor_boost = 1.5347641828430494 * rank_work_norm * successor_gate
    wait_norm = mad_normalize(ready_wait_time)
    wait_score = 1.0 - np.exp(-0.02436091936065679 * np.maximum(0.0, wait_norm))
    unc_norm = mad_normalize(uncertainty)
    uncertainty_amplifier = 0.40987791682209535 * unc_norm * np.clip(-slack, 0.0, np.inf)
    score = slack_norm + 0.6613912373005354 * duration_norm + energy_weight_adj * energy_norm - successor_boost + wait_score + uncertainty_amplifier
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
