import numpy as np
RULE_METADATA = {'structure_hash': 'e6e6f4fa6de454383693b834075055a073d2e59594efa7430ce2fdb89011e6ea', 'parameter_schema_hash': 'c5746cf086f5659694aa9b177fd6edf274f5234fe30ff201d761cd78ed17951b', 'best_parameter_hash': 'a5089d8299f222359bd7e6c2098f072addb02fb39498b4b264758a4a4c6956fe', 'best_parameters': {'epsilon': 7.150568498752202e-06, 'ddl_protection_threshold': 0.5484817456530623, 'critical_path_release_weight': 1.7818851815415861, 'risk_adjusted_energy_weight': 1.2159248943228493, 'wait_time_decay_exponent': 0.9767885871875142, 'slack_mad_scale': 1.2221357583142058, 'duration_risk_penalty': 1.0212594460851654, 'rank_slack_balance': 0.8584734995251974, 'slack_sigmoid_steepness': 3.727533976093524, 'tight_slack_uncertainty_coupling': 1.2377222747184242, 'normalized_wait_headroom_ratio': 0.8831299678829237}, 'optimizer_config_hash': 'bd16adfa68cb3d3c380e679bfefc37d2ca8416a7e2e3e8f05a56dd735c7a1d44', 'parameter_diagnostics_hash': 'ca3551dc7b204f556ccc21e78acab890349db74a8c511880105970b7b40fd7cc', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with one bounded structural improvement:
      - Replaces convex slack penalty with a *bounded sigmoid* over normalized slack,
        preserving strict DDL dominance (steep negative region) while avoiding numerical
        explosion and over-penalization of moderate lateness — directly addressing reflection.
      - Tight-slack uncertainty coupling is now *conditional and exclusive*: applied only when
        slack <= ddl_protection_threshold (not additive everywhere), decoupled from duration risk.
      - Wait-time anti-starvation scaled by normalized headroom ratio (not raw slack difference)
        for robustness across workflow scales.
      - All other components preserved from prior version's validated structure (MAD normalization,
        lexicographic gating, critical-path term, etc.), maintaining determinism and feasibility.
      - No new feature interactions beyond declared parameters; all numeric literals restricted to {-2,-1,0,1,2}.
    """
    eps = 7.150568498752202e-06
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
    duration_total = min_exec_time + min_comm_time + eps
    slack_abs_norm = mad_normalize(np.abs(slack))
    unc_norm = mad_normalize(uncertainty)
    dur_norm = mad_normalize(duration_total)
    slack_sigmoid_input = -slack_abs_norm
    slack_sigmoid = 1.0 / (1.0 + np.exp(-3.727533976093524 * slack_sigmoid_input))
    slack_penalty = slack_sigmoid * 1.2221357583142058 * (1.0 + unc_norm)
    tight_slack_mask = np.where(slack <= 0.5484817456530623, 1.0, 0.0)
    tight_slack_uncertainty_coupling = tight_slack_mask * uncertainty * slack_abs_norm * 1.2377222747184242
    duration_risk = duration_total * uncertainty * 1.0212594460851654
    critical_release_score = upward_rank * remaining_work * 1.7818851815415861
    slack_headroom_mask = np.where(slack > 0.5484817456530623, 1.0, 0.0)
    energy_norm = mad_normalize(min_incremental_energy)
    energy_term = slack_headroom_mask * 1.2159248943228493 * energy_norm
    wait_power = np.where(slack_headroom_mask > 0.0, ready_wait_time ** 0.9767885871875142, 0.0)
    wait_norm = mad_normalize(wait_power)
    wait_term = slack_headroom_mask * wait_norm * 0.8831299678829237
    rank_norm = mad_normalize(upward_rank)
    headroom_ratio = np.clip((slack - 0.5484817456530623) / (0.5484817456530623 + 1.0), 0.0, 1.0)
    weight_rank = 0.8584734995251974 + (1.0 - 0.8584734995251974) * (1.0 - headroom_ratio)
    rank_term = slack_headroom_mask * -rank_norm * weight_rank
    score = mad_normalize(slack_penalty) + mad_normalize(tight_slack_uncertainty_coupling) + mad_normalize(duration_risk) + mad_normalize(critical_release_score)
    score += energy_term + wait_term + rank_term
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
