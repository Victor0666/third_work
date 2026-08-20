import numpy as np
RULE_METADATA = {'structure_hash': 'dc95f855eb705b360efe12a3609ed997a255745aeff6f7c029048cfc74b8066e', 'parameter_schema_hash': 'db384522fecf786741cd1799d40910086628acdc482abf742fda08acc813de55', 'best_parameter_hash': '64b4b6a0fc2e1bcf594eebb0b97e472e13d36935eedfa9afdc73b9f81fb35ed3', 'best_parameters': {'epsilon': 0.005725499379473735, 'slack_risk_penalty': 0.758586780213085, 'slack_urgency_gain': 2.30194520600739, 'energy_efficiency_weight': 0.8755986376908887, 'criticality_weight': 2.089125411667057, 'duration_uncertainty_ratio': 0.8628143687785634, 'wait_decay_rate': 0.3132643858653781, 'uncertainty_slack_interaction': 4.9775758544102775, 'finfo_max_scale': 1156.7232441492067, 'ddl_protection_gate_width': 0.10963170845068261, 'load_successor_release_weight': 0.07002374414900062, 'host_load_sensitivity': 1.0668852092409788}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': '96a5abcc0be4c9e13aa5481e9f1bee950571f9aba1141c80ed5d6e1fc2362962', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule with smooth DDL protection gate, successor-release interaction, and host-load-aware energy scaling.
    
    Key structural mutations:
      - Replaces hard `slack_norm >= 0` gating with *smooth sigmoid gate* centered at slack=0 (controlled by ddl_protection_gate_width)
        → eliminates discontinuities, improves gradient stability for CMA-ES, and enables graceful degradation under marginal slack.
      - Adds *successor-release interaction*: prioritizes tasks whose completion releases many successors *only when system load is high* 
        (estimated via median-normalized remaining_work), addressing 'critical_path starvation' observed in replay failures.
      - Introduces *host-load-aware energy scaling*: applies power-law compression to energy_score when total remaining_work is high,
        preventing over-aggressive low-energy selection that delays critical path tasks under congestion.
      - Uses *median-based robust normalization* (instead of mean) for all feature scalings to suppress outlier influence.
      - All divisions guarded; all NaN/inf replaced; shape strictly enforced; deterministic.
    """
    eps = 0.005725499379473735
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    work = np.asarray(remaining_work, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)

    def normalize(x):
        x = np.asarray(x)
        abs_x = np.abs(x)
        med_abs = np.median(abs_x)
        scale = med_abs if np.all(np.isfinite(abs_x)) and med_abs > eps else eps
        return x / (scale + eps)
    slack_centered = slk / (np.median(np.abs(slk)) + eps)
    gate_smooth = 1.0 / (1.0 + np.exp(-slack_centered / 0.10963170845068261))
    slack_penalty = 0.758586780213085 * np.maximum(0.0, -slack_centered) - 2.30194520600739 * np.maximum(0.0, slack_centered)
    inv_energy = 1.0 / (energy + eps)
    energy_base = -0.8755986376908887 * normalize(inv_energy)
    load_factor = np.clip(normalize(work), 0.0, 2.0)
    energy_score = energy_base * (1.0 - load_factor ** 1.0668852092409788)
    rank_score = -2.089125411667057 * normalize(rank + eps) * gate_smooth
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.8628143687785634 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_sat = 1.0 - np.exp(-0.3132643858653781 * wait)
    wait_score = -normalize(wait_sat + eps)
    unc_slack_interaction = 4.9775758544102775 * uncert_norm * np.maximum(0.0, -slack_centered)
    successor_release_boost = 0.07002374414900062 * normalize(rank + eps) * (1.0 - gate_smooth) * load_factor
    score = slack_penalty + energy_score + rank_score + dur_score + wait_score + unc_slack_interaction + successor_release_boost
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 1156.7232441492067
    min_safe = -finfo.max / 1156.7232441492067
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
