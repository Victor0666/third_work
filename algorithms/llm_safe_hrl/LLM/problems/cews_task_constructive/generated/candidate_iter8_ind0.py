import numpy as np
RULE_METADATA = {'structure_hash': '73c22b8b9301680c9c48ab1dc84db8b88ae1ada8edcad35bb52fc7430dbe17e4', 'parameter_schema_hash': '2ca90c76e6cb3f265f3fd110422ecd7bf6399c898e2edb378351ca2d4be66fc4', 'best_parameter_hash': '79c047e71e5ec73a6e64774d651a53681eb8c7755d6864d8ecb37e2e2452a636', 'best_parameters': {'epsilon': 0.0012924621180095406, 'ddl_protection_gate_threshold': 0.6450289443649315, 'energy_duration_ratio_weight': 0.7989952383824339, 'successor_bottleneck_coupling': 0.7749374049414961, 'iqr_low_percentile': 33.53491305398464, 'iqr_high_percentile': 76.65735175520737, 'wait_ramp_scale': 5.4050797311791765, 'urgency_amplification_threshold': 1.9352095098401003, 'urgency_amplification_factor': 1.1473552052208524}, 'optimizer_config_hash': '087d89d0b2174a4ef39ab75b5292a4f2a6641d337c4dd6a2bb86ea7b8b713284', 'parameter_diagnostics_hash': '231d8607992e71e2e278071cf35048869448e64b38312204d0dd843683c6c69d', 'optimizer_seed': 0, 'training_seeds': [0, 1, 2], 'validation_seeds': [3, 4]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's stability with Parent 1's hard-DDL enforcement.
    
    Key structural improvements:
      - Introduces *adaptive urgency amplification*: when slack < urgency_amplification_threshold,
        urgency is multiplied by urgency_amplification_factor — a clean, interpretable, bounded
        mechanism to enforce hard deadlines without fragile sigmoid or binary gating.
      - Retains Parent 2's robust bottleneck term: duration × (upward_rank × remaining_work),
        confirmed superior in unblocking critical successors.
      - Keeps linear urgency mapping (sliding [-1,1] via median-normalized slack) for monotonicity,
        but adds amplification only where needed — avoids over-suppression of non-critical tasks.
      - Drops all inactive features (critical-rank gate, release cap, feasibility guard mask) per analysis.
      - Uses elite-tuned IQR percentiles (35/77) for outlier-resilient normalization.
      - DDL-protection gate now uses *median-relative slack* AND *scaled uncertainty*, matching Parent 2's high-confidence logic.
      - Ready wait time remains linear ramp for fairness, anti-starvation, and simplicity.
      - All operations guarded against NaN/inf/zero; deterministic and finite output.
    """
    eps = 0.0012924621180095406
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    def iqr_normalize(x):
        x = np.copy(x)
        q_low = np.percentile(x, 33.53491305398464)
        q_high = np.percentile(x, 76.65735175520737)
        iqr = q_high - q_low
        center = np.median(x)
        denom = iqr if iqr > eps else np.max(np.abs(x - center)) + eps
        return (x - center) / (denom + eps)
    median_slack = np.median(slack)
    urgency_base = np.clip(-slack / (np.abs(median_slack) + eps), -1.0, 1.0)
    amplification_mask = (slack < 1.9352095098401003).astype(float)
    urgency_amplified = urgency_base * (1.0 + 1.1473552052208524 * amplification_mask)
    norm_urgency = iqr_normalize(urgency_amplified)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / duration
    norm_energy_eff = iqr_normalize(energy_per_duration)
    bottleneck_pressure = duration * (upward_rank * remaining_work + eps)
    norm_bottleneck = iqr_normalize(bottleneck_pressure)
    wait_ramp = np.clip(ready_wait_time / 5.4050797311791765, 0.0, 1.0)
    norm_wait = iqr_normalize(wait_ramp)
    max_uncertainty = np.max(uncertainty) if N > 0 else 1.0
    ddl_protection_active = ((slack < median_slack) & (uncertainty > 0.6450289443649315 * max_uncertainty)).astype(float)
    norm_uncertainty = iqr_normalize(uncertainty)
    score = norm_urgency + 0.7749374049414961 * norm_bottleneck + 0.7989952383824339 * norm_energy_eff - norm_wait + ddl_protection_active * norm_uncertainty
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
