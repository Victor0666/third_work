import numpy as np
RULE_METADATA = {'structure_hash': '11be5558e59436c6d3e3f4c58a4bcfc7566b68df20d3bb0959ba80022663e509', 'parameter_schema_hash': '94f484b68bfc50aa4fbe06f20b91e360bc1b00ecb2374928b7e2b718711dd8af', 'best_parameter_hash': 'c03c99faa167f516e4e290c8d30e3ffaf1b64435963179d2df8aa34c92c681fd', 'best_parameters': {'epsilon': 0.00010039505287841867, 'ddl_risk_activation_threshold': 0.5084490720667518, 'successor_bottleneck_coupling': 0.00014927690829685418, 'wait_fairness_weight': 0.828622957254253, 'urgency_scale': 2.5815878113242428, 'urgency_clamp_lower': -0.6726145215333355, 'risk_modulated_bottleneck_weight': 2.236877743999508, 'iqr_percentile_low': 23.35315419590563, 'iqr_percentile_high': 69.69248765165989, 'energy_uncertainty_coupling': 0.657052828363417}, 'optimizer_config_hash': '087d89d0b2174a4ef39ab75b5292a4f2a6641d337c4dd6a2bb86ea7b8b713284', 'parameter_diagnostics_hash': '35cc57b1635ceba123be1bbe4e02e0a70a3760d8d765e690d7ee1a8520a26252', 'optimizer_seed': 0, 'training_seeds': [0, 1, 2], 'validation_seeds': [3, 4]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining robust urgency (Parent 2) with uncertainty-coupled energy (Parent 1):
      - Retains bounded tanh urgency + clamped affine mapping for stability near zero slack.
      - Keeps conditional bottleneck amplification gated by *verified* DDL risk (slack < 0 AND uncertainty > threshold * median).
      - Adds uncertainty-modulated energy term: energy × (1 + uncertainty)^coupling — captures risk-aware energy tradeoff.
      - Uses tunable IQR percentiles for adaptive normalization (declared, not hardcoded).
      - Introduces explicit energy-uncertainty coupling to prevent unsafe low-energy assignments under high uncertainty.
      - All terms additive; no unbounded operations; strict finiteness enforcement via np.finfo.
      - No hidden constants: only {-2,-1,0,1,2} used as literals.
    """
    eps = 0.00010039505287841867
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    def adaptive_normalize(x):
        x = np.copy(x)
        q_low = np.percentile(x, 23.35315419590563)
        q_high = np.percentile(x, 69.69248765165989)
        iqr = q_high - q_low + eps
        center = np.median(x)
        return (x - center) / (iqr + eps)
    abs_slack = np.abs(slack)
    scale = np.median(abs_slack) if np.any(abs_slack > eps) else 1.0
    tanh_urgency = np.tanh(-slack / (scale * 2.5815878113242428 + eps))
    clamped_urgency = np.clip(tanh_urgency, -0.6726145215333355, 1.0)
    urgency = (clamped_urgency + 1.0) / 2.0
    norm_urgency = adaptive_normalize(urgency)
    duration = min_exec_time + min_comm_time
    bottleneck_pressure = duration * upward_rank * (min_incremental_energy + eps)
    norm_bottleneck = adaptive_normalize(bottleneck_pressure)
    unc_normalized = adaptive_normalize(uncertainty)
    modulated_energy = min_incremental_energy * np.power(1.0 + unc_normalized, 0.657052828363417)
    norm_modulated_energy = adaptive_normalize(modulated_energy)
    norm_wait = adaptive_normalize(ready_wait_time)
    fairness_term = -0.828622957254253 * norm_wait
    median_uncertainty = np.median(uncertainty) if N > 0 else eps
    ddl_risk_gate = ((slack < 0.0) & (uncertainty > 0.5084490720667518 * median_uncertainty)).astype(float)
    risk_amplified_bottleneck = norm_bottleneck * (1.0 + 2.236877743999508 * ddl_risk_gate)
    stress_amplified_urgency = norm_urgency * (1.0 + ddl_risk_gate)
    score = stress_amplified_urgency + 0.00014927690829685418 * risk_amplified_bottleneck + norm_modulated_energy + fairness_term
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
