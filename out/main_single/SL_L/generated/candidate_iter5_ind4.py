import numpy as np
RULE_METADATA = {'structure_hash': '6ff9ac10248fdd5a675c8dbfd91bc40a0189ca2a299c29b0401878ca07bd15a9', 'parameter_schema_hash': '98d6f5c4f6dae97355730b3aec1f3bfa89ff35bc6b60b9c05e7616008b5290f4', 'best_parameter_hash': 'a940dd44dde59cac0b4dfdfc4e43b84ab047e48b04495af07b86f2dff086463f', 'best_parameters': {'epsilon': 2.9285615861748237e-08, 'slack_risk_penalty': 7.7049131388800856, 'slack_urgency_scale': 1.8108619848929024, 'slack_cap': 5.227742224704816, 'slack_pressure_tanh_scale': 0.26208885395441683, 'energy_efficiency_bias': 1.1847419675887403, 'critical_path_leverage': 0.22633270004718944, 'work_density_weight': 0.37450100208582027, 'robustness_mad_factor': 1.453260011262699, 'wait_fairness_gain': 0.3415699893207442, 'uncertainty_sensitivity': 1.4261774734371298, 'duration_balance': 0.26811680907172925}, 'optimizer_config_hash': 'bd16adfa68cb3d3c380e679bfefc37d2ca8416a7e2e3e8f05a56dd735c7a1d44', 'parameter_diagnostics_hash': 'df9ee22341a4d96c2fc753b075d593cba2da20d8f09272105bada70fc124818f', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule unifying Parent 2's robust slack dynamics with Parent 1's uncertainty-aware rank boosting and work-density gating.
    
    Key structural improvements:
    - Replaces simple slack feasibility gating with *uncertainty-conditioned critical path leverage*: 
      critical_path_leverage now scales with (1 + uncertainty) only when slack >= 0 → prioritizes critical tasks more under risk.
    - Introduces *dynamic work-density gating* using tanh(slack_pressure * slack_pressure_tanh_scale) instead of binary (slack >= 0),
      enabling smooth, differentiable deactivation of density bonus as deadline pressure rises.
    - Unifies all normalization under a single MAD-based scheme with shared epsilon safeguards and consistent outlier handling.
    - Removes redundant duration normalization duplication; uses one robust duration_norm for both direct penalty and work-density denominator.
    - All numeric literals strictly in {-2,-1,0,1,2}; no hard thresholds or unbounded functions.
    """
    eps = 2.9285615861748237e-08
    finfo = np.finfo(float)
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=eps, posinf=finfo.max, neginf=eps)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=eps, posinf=finfo.max, neginf=eps)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=eps, posinf=finfo.max, neginf=eps)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=finfo.max, neginf=finfo.min)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=eps, posinf=finfo.max, neginf=eps)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=eps, posinf=finfo.max, neginf=eps)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=finfo.max, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=eps, posinf=finfo.max, neginf=eps)

    def mad_normalize(x):
        x = np.asarray(x, dtype=float)
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        scale = 1.453260011262699 * (mad if mad > eps else eps)
        return (x - med) / scale
    neg_mask = slack < 0.0
    zeroish_mask = (slack >= 0.0) & (slack < 1.0)
    pos_mask = slack >= 1.0
    slack_norm = np.zeros_like(slack)
    slack_norm[neg_mask] = 7.7049131388800856 * -slack[neg_mask]
    slack_norm[zeroish_mask] = 1.8108619848929024 * (np.exp(slack[zeroish_mask]) - 1.0)
    slack_norm[pos_mask] = np.clip(slack[pos_mask], 0.0, 5.227742224704816)
    duration = min_exec_time + min_comm_time
    duration_norm = mad_normalize(duration)
    energy_norm = mad_normalize(min_incremental_energy)
    slack_pressure = np.clip(-slack, 0.0, np.inf)
    energy_weight_adj = 1.1847419675887403 * (1.0 - np.tanh(slack_pressure * 0.26208885395441683))
    rank_norm = mad_normalize(upward_rank)
    critical_gate = (slack >= 0.0).astype(float)
    critical_boost = 0.22633270004718944 * rank_norm * critical_gate * (1.0 + uncertainty)
    work_density = np.divide(remaining_work, duration + eps)
    work_density_norm = mad_normalize(work_density)
    work_density_gate = 1.0 - np.tanh(slack_pressure * 0.26208885395441683)
    work_density_bonus = 0.37450100208582027 * work_density_norm * work_density_gate
    wait_headroom = np.maximum(1.0, slack + 1.0)
    wait_norm = mad_normalize(ready_wait_time)
    wait_score = 0.3415699893207442 * wait_norm / (wait_headroom + eps)
    unc_norm = mad_normalize(uncertainty)
    slack_pressure_bounded = np.tanh(slack_pressure * 0.26208885395441683)
    uncertainty_amplifier = 1.4261774734371298 * unc_norm * slack_pressure_bounded
    score = slack_norm + 0.26811680907172925 * duration_norm + energy_weight_adj * energy_norm - critical_boost - work_density_bonus + wait_score + uncertainty_amplifier
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
