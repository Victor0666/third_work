import numpy as np
RULE_METADATA = {'structure_hash': '9f81587f15ea006d6dad4412a03f3426886b317374c183cede3fb18995b0f2dc', 'parameter_schema_hash': '521366f6b2eed40aab59fa824d1d34240125ff83dcbbe8c38a0efaf48a41a37c', 'best_parameter_hash': '4b060816b29eb87df23ba6de1455a25faa1a7f3a847a7aca1da6d69eb75093c5', 'best_parameters': {'epsilon': 1.3726705598198766e-08, 'slack_penalty_exponent': 2.4908559115494837, 'criticality_boost': 3.433096452447333, 'energy_efficiency_ratio_weight': 0.6380280146045496, 'uncertainty_slack_coupling': 0.023690579371838612, 'rank_slack_balance': 0.8885160173761492, 'duration_risk_penalty': 0.03104131607524874, 'energy_uncertainty_interaction': 0.09942468583484924, 'ddl_protection_threshold': 0.037194512154484126, 'upward_rank_remaining_work_interaction': 1.373176302162257, 'host_load_conditional_gate': 0.024631436853052843}, 'optimizer_config_hash': 'bd16adfa68cb3d3c380e679bfefc37d2ca8416a7e2e3e8f05a56dd735c7a1d44', 'parameter_diagnostics_hash': 'ef67afcd3e33c02ccf4bf029c2b9f1e9a0ff912ea7080c515538771891c6a4a7', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule incorporating three key structural improvements:
      - Replaces fragile exponentiated wait-slack coupling with robust bounded linear denominator: (1 + clip(slack, 0, 2))
      - Replaces tanh-based uncertainty gate with clipped linear gate: clip(uncertainty, 0, 1), eliminating unstable gradients
      - Restores host-load conditional gate to suppress premature energy optimization on marginal tasks
      - Introduces unified 'urgency density' feature: crit_path_urgency / (min_exec_time + min_comm_time + eps), emphasizing fast-release of critical work
      - All MAD-normalized components strictly clipped to [-2, 2] for guaranteed robustness at N=1 and sparse sets
      - Final score maintains strict lexicographic DDL safety: violation terms dominate; non-DDL terms gated by slack_headroom_mask AND host_load_gate
    """
    eps = 1.3726705598198766e-08
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

    def mad_normalize(x):
        x = np.asarray(x, dtype=float)
        if N == 1:
            return np.zeros_like(x, dtype=float)
        med = np.median(x)
        mad = np.median(np.abs(x - med)) + eps
        normalized = (x - med) / mad
        return np.clip(normalized, -2.0, 2.0)
    slack_score = np.where(slack < 0, (-slack) ** 2.4908559115494837, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 0.023690579371838612
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 0.03104131607524874
    crit_path_urgency = upward_rank * remaining_work * 1.373176302162257
    duration_total = min_exec_time + min_comm_time + eps
    urgency_density = crit_path_urgency / duration_total
    slack_headroom_mask = np.where(slack > 0.037194512154484126, 1.0, 0.0)
    slack_clipped = np.clip(slack, 0.0, 2.0)
    wait_denom = 1.0 + slack_clipped
    wait_base = ready_wait_time / wait_denom
    wait_score = wait_base * np.sign(slack + eps)
    energy_per_sec = min_incremental_energy / duration_total
    energy_per_sec_med = np.median(energy_per_sec) if N > 1 else np.mean(energy_per_sec)
    host_load_gate = np.where(energy_per_sec < energy_per_sec_med * 0.024631436853052843, 1.0, 0.0)
    energy_per_work = min_incremental_energy / (remaining_work + eps)
    energy_eff_score = mad_normalize(energy_per_work)
    slack_lb = -2.0
    slack_ub = 2.0
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    weight_rank = 0.8885160173761492 + (1.0 - 0.8885160173761492) * (1.0 - slack_scaled)
    rank_score = -mad_normalize(upward_rank) * weight_rank
    unc_linear_gate = np.clip(uncertainty, 0.0, 1.0)
    energy_norm = mad_normalize(min_incremental_energy)
    unc_norm = mad_normalize(uncertainty)
    energy_uncertainty_score = 0.09942468583484924 * energy_norm * unc_norm * unc_linear_gate
    slack_norm = mad_normalize(slack_score)
    unc_slack_norm = mad_normalize(unc_slack_coupling)
    duration_norm = mad_normalize(duration_risk)
    crit_path_norm = mad_normalize(crit_path_urgency)
    urgency_density_norm = mad_normalize(urgency_density)
    wait_norm = mad_normalize(wait_score)
    score = slack_norm + unc_slack_norm + duration_norm
    score += crit_path_norm
    score += urgency_density_norm
    score += slack_headroom_mask * host_load_gate * (0.6380280146045496 * energy_eff_score + rank_score + energy_uncertainty_score + wait_norm)
    is_ddl_pressure = slack <= 0
    critical_gate = np.where(is_ddl_pressure, 3.433096452447333, 1.0)
    score = score * critical_gate
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
