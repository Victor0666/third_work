import numpy as np
RULE_METADATA = {'structure_hash': 'f0110ffbac3a4b2a3d1c8ec35547bffe147fc83da0c4ebabf40204fe008fa2d5', 'parameter_schema_hash': '940a692f3e20a96b8c4b187d345705597e2795120c8e15607b8409ad94dd2d1f', 'best_parameter_hash': '67ab655431cdd3537bacf59cc02f7f0b67c7207440aae1bff089af41a5aad8b1', 'best_parameters': {'epsilon': 1.445152195907635e-05, 'energy_duration_ratio_weight': 1.085517154052042, 'successor_bottleneck_coupling': 0.046964842715016705, 'uncertainty_dispersion_scale': 0.5139727812790377, 'urgency_cap_exponent': 0.6760061085264643, 'bottleneck_uncertainty_amplification': 1.0967607240658688, 'wait_saturation_scale': 5.333660906155253, 'slack_gradient_sensitivity': 1.2798005149580058}, 'optimizer_config_hash': '087d89d0b2174a4ef39ab75b5292a4f2a6641d337c4dd6a2bb86ea7b8b713284', 'parameter_diagnostics_hash': 'e02b3fc769c60575261d565d2cabed5380f4a1f2cf4fabd5dc76f0eeb165bbf9', 'optimizer_seed': 0, 'training_seeds': [0, 1, 2], 'validation_seeds': [3, 4]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with slack-gradient modulation:
      - Replaces binary DDL-protection gating with *continuous slack-gradient sensitivity*:
        bottleneck and energy weights scaled by normalized local slack slope (median - slack) / (std(slack)+eps),
        preserving signal strength under skewed but non-critical slack distributions.
      - Retains pre-normalized `neg_slack` as dominant hard-DDL penalty — no distortion, no amplification.
      - Keeps bounded sigmoid wait saturation for robust anti-starvation.
      - All normalization uses uncertainty-aware `adaptive_normalize()` with fallback to range.
      - Final composition: hard-DDL > urgency > modulated bottleneck > modulated energy > fairness.
      - No branching; all operations vectorized and finite-valued.
    """
    eps = 1.445152195907635e-05
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
        center = np.median(x) if N > 0 else 0.0
        unc_std = np.std(uncertainty) if N > 1 else eps
        dispersion = 0.5139727812790377 * (unc_std + eps)
        fallback_range = np.max(x) - np.min(x) if N > 0 else eps
        denom = np.where(dispersion > eps, dispersion, fallback_range)
        return (x - center) / (denom + eps)
    neg_slack = np.clip(-slack, 0.0, None)
    median_slack = np.median(slack) if N > 0 else 0.0
    urgency_linear = np.clip(median_slack - slack, 0.0, 2.0)
    non_neg_slack = np.clip(slack, 0.0, None)
    max_non_neg_slack = np.max(non_neg_slack) if N > 0 else eps
    urgency_cap = np.power(max_non_neg_slack + eps, 0.6760061085264643)
    urgency_linear = np.minimum(urgency_linear, urgency_cap)
    norm_urgency = adaptive_normalize(urgency_linear)
    slack_std = np.std(slack) if N > 1 else eps
    slack_gradient = (median_slack - slack) / (slack_std + eps)
    modulation_factor = np.clip(slack_gradient, 0.0, 2.0) ** 1.2798005149580058
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / (duration + eps)
    norm_energy_eff = adaptive_normalize(energy_per_duration)
    bottleneck_base = duration * upward_rank * remaining_work * (1.0 + urgency_linear + eps)
    unc_normalized = adaptive_normalize(uncertainty)
    bottleneck_pressure = bottleneck_base * np.power(1.0 + unc_normalized, 1.0967607240658688)
    norm_bottleneck = adaptive_normalize(bottleneck_pressure)
    wait_scaled = ready_wait_time / (5.333660906155253 + eps)
    wait_clipped = np.clip(wait_scaled, -2.0, 2.0)
    wait_saturation = 1.0 / (1.0 + np.exp(-wait_clipped))
    norm_wait = adaptive_normalize(wait_saturation)
    score = neg_slack + norm_urgency + modulation_factor * 0.046964842715016705 * norm_bottleneck + modulation_factor * 1.085517154052042 * norm_energy_eff - norm_wait
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
