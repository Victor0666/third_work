import numpy as np
RULE_METADATA = {'structure_hash': 'f17da925c1dce672abace846a0b2685ace2453455a49e6678a2953a8ddf9eb1f', 'parameter_schema_hash': 'c23dcf598092685e66a6193b2760d3267ac87c9c1e608f929c656580cfb1c709', 'best_parameter_hash': '4fab2a64a26d1097bdb4fd06ff223ecca9cc09216c16a7860c0c24dfbcde86ac', 'best_parameters': {'epsilon': 3.251397549805471e-05, 'ddl_risk_activation_threshold': 0.7873459738604951, 'successor_bottleneck_coupling': 0.24218632716728436, 'wait_fairness_weight': 0.8586181506206931, 'urgency_scale': 0.17185973477774866, 'urgency_clamp_lower': -0.6904606820490296, 'risk_modulated_bottleneck_weight': 1.5105551454427766, 'iqr_percentile_low': 18.71767440208049, 'iqr_percentile_high': 70.6443294310457, 'energy_uncertainty_coupling': 0.2939970171271579}, 'optimizer_config_hash': '087d89d0b2174a4ef39ab75b5292a4f2a6641d337c4dd6a2bb86ea7b8b713284', 'parameter_diagnostics_hash': 'bd9b778886c2e553e63563d8fa23f92e57f27921daba55f87336fd9229dbb4a0', 'optimizer_seed': 0, 'training_seeds': [0, 1, 2], 'validation_seeds': [3, 4]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's robust urgency & conditional bottleneck with Parent 1's explicit energy-uncertainty coupling:
      - Retains tanh-based urgency with clamped affine mapping for smooth deadline proximity response
      - Keeps tunable IQR normalization for adaptive dispersion across heterogeneous workloads
      - Preserves conditional bottleneck amplification gated by *verified* DDL risk (slack < 0 AND uncertainty > threshold * median)
      - Adds novel energy_uncertainty_coupling: modulates incremental energy by normalized uncertainty to discourage high-risk low-energy VMs
      - Removes redundant power-law DDL penalty (Parent 1) — tanh + gating is more stable and interpretable
      - All terms additive, bounded, and finite; no hidden constants beyond {-2,-1,0,1,2}
      - Fairness remains subtractive to prevent starvation
      - Normalization avoids division-by-zero via eps and IQR fallback
    """
    eps = 3.251397549805471e-05
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
        if N == 0:
            return np.zeros_like(x)
        q_low = np.percentile(x, 18.71767440208049)
        q_high = np.percentile(x, 70.6443294310457)
        iqr = q_high - q_low + eps
        center = np.median(x)
        return (x - center) / iqr
    abs_slack = np.abs(slack)
    scale = np.median(abs_slack) if np.any(abs_slack > eps) else 1.0
    tanh_urgency = np.tanh(-slack / (scale * 0.17185973477774866 + eps))
    clamped_urgency = np.clip(tanh_urgency, -0.6904606820490296, 1.0)
    urgency = (clamped_urgency + 1.0) / 2.0
    norm_urgency = adaptive_normalize(urgency)
    duration = min_exec_time + min_comm_time
    bottleneck_pressure = duration * upward_rank * (min_incremental_energy + eps)
    norm_bottleneck = adaptive_normalize(bottleneck_pressure)
    unc_normalized = adaptive_normalize(uncertainty)
    risk_adjusted_energy = min_incremental_energy * (1.0 + 0.2939970171271579 * np.abs(unc_normalized))
    norm_energy = adaptive_normalize(risk_adjusted_energy)
    norm_wait = adaptive_normalize(ready_wait_time)
    fairness_term = -0.8586181506206931 * norm_wait
    median_uncertainty = np.median(uncertainty) if N > 0 else eps
    ddl_risk_gate = ((slack < 0.0) & (uncertainty > 0.7873459738604951 * median_uncertainty)).astype(float)
    risk_amplified_bottleneck = norm_bottleneck * (1.0 + 1.5105551454427766 * ddl_risk_gate)
    stress_amplified_urgency = norm_urgency * (1.0 + ddl_risk_gate)
    score = stress_amplified_urgency + 0.24218632716728436 * risk_amplified_bottleneck + norm_energy + fairness_term
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
