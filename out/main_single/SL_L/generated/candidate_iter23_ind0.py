import numpy as np
RULE_METADATA = {'structure_hash': 'f69f76b62a14881749881287cab06946f6ecd5d2d16da2414da698ef9bf0a6f1', 'parameter_schema_hash': '19f9d1aefbe61055bd0b159486d7c6f897aad62903975d78bc2d65e8cb6e0dda', 'best_parameter_hash': '88aab71979c4a5ef3ee428863328ead5a5870f9fd768de0c2110c5e9fa28850f', 'best_parameters': {'epsilon': 1.0483303447892675e-09, 'slack_penalty_exponent': 1.612303886449734, 'energy_efficiency_ratio_weight': 0.10281793867206131, 'uncertainty_slack_coupling': 1.5867807440115165, 'rank_slack_balance': 0.9316228345675515, 'duration_risk_penalty': 0.9201955539563187, 'energy_uncertainty_interaction': 0.25128003917641567, 'uncertainty_sigmoid_steepness': 1.2408921648957878, 'ddl_protection_threshold': 1.7790758228902928, 'upward_rank_remaining_work_interaction': 0.2533897891058906, 'host_load_conditional_gate': 1.0864184757244122, 'feasibility_mask_penalty': 339904.178946424}, 'optimizer_config_hash': 'bd16adfa68cb3d3c380e679bfefc37d2ca8416a7e2e3e8f05a56dd735c7a1d44', 'parameter_diagnostics_hash': 'caafbfd4908dcfff0dc3d932e7bb381aac2d0d4d5947a4c5f0bf7241857ac4a2', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's hard feasibility enforcement and task-wise normalization
    with Parent 1's starvation mitigation and refined critical-path signaling.
    
    Key structural improvements:
      - Retains Parent 2's *hard feasibility mask* and *task-wise MAD normalization* for robustness
      - Integrates Parent 1's *wait_starvation_penalty* as unconditional but risk-weighted term
      - Replaces simple critical_release with a *two-tier criticality signal*: 
          (a) raw upward_rank * remaining_work (for global critical path), and 
          (b) normalized slack-aware rank prioritization (for local urgency)
      - Introduces *dynamic wait-gating*: starvation reward is amplified under tight slack via smooth clip
      - Uses *clipped, bounded normalization* (-2 to 2) in all task-wise transforms to prevent outlier skew
      - All numeric literals strictly limited to {-2,-1,0,1,2}
    """
    eps = 1.0483303447892675e-09
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

    def task_wise_mad_normalize(x):
        x = np.asarray(x, dtype=float)
        if N == 1:
            return np.zeros_like(x, dtype=float)
        med = np.median(x)
        mad = np.median(np.abs(x - med)) + eps
        centered = x - med
        normalized = centered / mad
        return np.clip(normalized, -2.0, 2.0)
    duration_total = min_exec_time + min_comm_time + eps
    abs_slack = np.abs(slack) + eps
    slack_norm = task_wise_mad_normalize(slack)
    abs_slack_norm = task_wise_mad_normalize(abs_slack)
    uncertainty_norm = task_wise_mad_normalize(uncertainty)
    duration_norm = task_wise_mad_normalize(duration_total)
    energy_norm = task_wise_mad_normalize(min_incremental_energy)
    rank_norm = task_wise_mad_normalize(upward_rank)
    work_norm = task_wise_mad_normalize(remaining_work)
    wait_norm = task_wise_mad_normalize(ready_wait_time)
    feasibility_mask = np.where(slack < 0, 339904.178946424, 0.0)
    slack_score = np.where(slack < 0, (-slack) ** 1.612303886449734, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 1.5867807440115165
    duration_risk = duration_total * uncertainty * 0.9201955539563187
    critical_release_raw = upward_rank * remaining_work * 0.2533897891058906
    critical_release_score = task_wise_mad_normalize(critical_release_raw)
    slack_urgency_weight = np.clip(1.0 + slack_norm, 0.0, 2.0)
    local_rank_score = -rank_norm * slack_urgency_weight
    ddl_safe_mask = np.where(slack > 1.7790758228902928, 1.0, 0.0)
    duration_median = np.median(duration_total) if N > 1 else np.mean(duration_total)
    host_margin_sufficient = np.where(duration_total <= 1.0864184757244122 * duration_median, 1.0, 0.0)
    energy_per_sec = min_incremental_energy / (duration_total + eps)
    energy_eff_score = task_wise_mad_normalize(energy_per_sec)
    weight_rank = 0.9316228345675515 + (1.0 - 0.9316228345675515) * (1.0 - np.clip(slack_norm, 0.0, 1.0))
    rank_score = -rank_norm * weight_rank
    unc_sigmoid = 1.0 / (1.0 + np.exp(-1.2408921648957878 * (uncertainty - 1.0)))
    energy_uncertainty_score = 0.25128003917641567 * energy_norm * uncertainty_norm * unc_sigmoid
    wait_gate = np.clip(1.0 + np.abs(slack_norm), 0.0, 2.0)
    wait_score = wait_norm * wait_gate
    score = feasibility_mask + abs_slack_norm + uncertainty_norm + duration_norm + slack_score + unc_slack_coupling + duration_risk + critical_release_score + local_rank_score
    score += ddl_safe_mask * host_margin_sufficient * (0.10281793867206131 * energy_eff_score + rank_score + energy_uncertainty_score + wait_score)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
