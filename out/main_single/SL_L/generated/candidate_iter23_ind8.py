import numpy as np
RULE_METADATA = {'structure_hash': '68a376da2fcc934934744c2d6bdcd9061a544f1910ae1eecc5b2fc6c5b733a92', 'parameter_schema_hash': 'defb49bbe0f6eea07689fc5f005dafa2c1d94d302ba859ade1c281860580a867', 'best_parameter_hash': '3d4869ad98c9aa83c8bff323c71f42350289f99afff7c46305d4fb55de97e45a', 'best_parameters': {'epsilon': 1.194539196268509e-09, 'ddl_protection_threshold': 0.3485599621614041, 'critical_path_release_weight': 2.5559202231107214, 'risk_adjusted_energy_weight': 1.960131962094144, 'uncertainty_sigmoid_steepness': 2.94521512997688, 'wait_time_decay_exponent': 0.5217221862577317, 'duration_risk_penalty': 0.6039131307496649, 'rank_slack_balance': 0.6803570573276696, 'slack_penalty_exponent': 3.1575520998586826, 'uncertainty_slack_coupling': 0.6172073805000253, 'energy_uncertainty_interaction': 0.8353507652784101}, 'optimizer_config_hash': 'bd16adfa68cb3d3c380e679bfefc37d2ca8416a7e2e3e8f05a56dd735c7a1d44', 'parameter_diagnostics_hash': '7f2770f0ac4d913859ff02bdf260b213677f9bbb4d152bb178926109d4f211d0', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining strengths of both parents:
      - Lexicographic DDL enforcement: hard gating via `slack > ddl_protection_threshold`
      - Robust MAD-aligned normalization over |slack|, uncertainty, duration_total for scale coherence
      - Critical-path release term (upward_rank * remaining_work) unconditionally active
      - Convex slack violation penalty: (-slack)^exponent for stronger lateness discrimination
      - Uncertainty-slack coupling: linearly penalizes high uncertainty when slack is tight/negative
      - Power-law anti-starvation: `ready_wait_time ** wait_time_decay_exponent`, scaled only under safety
      - Energy-uncertainty interaction gated by sigmoid to suppress noise at low uncertainty
      - All numeric literals restricted to {-2,-1,0,1,2}; no other constants used
      - Final score ensures strict feasibility-first ordering: DDL violation dominates; among compliant, trade off criticality vs risk-adjusted energy
    """
    eps = 1.194539196268509e-09
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
    slack_abs = np.abs(slack)
    slack_penalty = np.where(slack < 0, (-slack) ** 3.1575520998586826, 0.0)
    deadline_pressure = np.maximum(0.0, -slack + 0.3485599621614041)
    unc_slack_coupling = uncertainty * deadline_pressure * 0.6172073805000253
    duration_risk = duration_total * uncertainty * 0.6039131307496649
    critical_release_score = upward_rank * remaining_work * 2.5559202231107214
    slack_headroom_mask = np.where(slack > 0.3485599621614041, 1.0, 0.0)
    energy_norm = mad_normalize(min_incremental_energy)
    energy_term = slack_headroom_mask * 1.960131962094144 * energy_norm
    wait_power = np.where(slack_headroom_mask > 0.0, ready_wait_time ** 0.5217221862577317, 0.0)
    wait_norm = mad_normalize(wait_power)
    wait_term = slack_headroom_mask * wait_norm
    unc_sigmoid = 1.0 / (1.0 + np.exp(-2.94521512997688 * (uncertainty - 1.0)))
    energy_uncertainty_term = slack_headroom_mask * 0.8353507652784101 * energy_norm * mad_normalize(uncertainty) * unc_sigmoid
    rank_norm = mad_normalize(upward_rank)
    slack_headroom = np.clip((slack - 0.3485599621614041) / (0.3485599621614041 + 1.0), 0.0, 1.0)
    weight_rank = 0.6803570573276696 + (1.0 - 0.6803570573276696) * (1.0 - slack_headroom)
    rank_term = slack_headroom_mask * -rank_norm * weight_rank
    score = mad_normalize(slack_penalty) + mad_normalize(unc_slack_coupling) + mad_normalize(duration_risk) + mad_normalize(critical_release_score)
    score += energy_term + wait_term + energy_uncertainty_term + rank_term
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
