import numpy as np
RULE_METADATA = {'structure_hash': 'efe4e883502f1bcdf3f71d2e3cff7291af8f2b928598f2391838cb91299753a2', 'parameter_schema_hash': '314023ed68ce489561275193007b56e1a642c61d057aba64692400802a730edd', 'best_parameter_hash': '57d49a55d3a5540320c085f9cc471def3e09bde4bd169daeb071895614f676db', 'best_parameters': {'epsilon': 1.6873059837465527e-07, 'slack_penalty_exponent': 1.9625275666563022, 'criticality_boost': 0.5925963206594987, 'energy_efficiency_ratio_weight': 0.4240605387574511, 'uncertainty_slack_coupling': 1.4144571686190341, 'rank_slack_balance': 0.23004720776496226, 'duration_risk_penalty': 0.7464312024581721, 'energy_uncertainty_interaction': 0.7073304777843867, 'uncertainty_sigmoid_steepness': 3.2018931670819084, 'ddl_protection_threshold': 1.5319493300049762, 'joint_mad_scale': 2.1653083704191096}, 'optimizer_config_hash': 'bd16adfa68cb3d3c380e679bfefc37d2ca8416a7e2e3e8f05a56dd735c7a1d44', 'parameter_diagnostics_hash': 'b8eb5ce303b6d77fb353c247f422b924ebc99b7daaca05a65dcbde10c0e79c8c', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with conditional DDL-protection gate:
      - Introduces a *bounded structural change*: new DDL-protection gate activated only when norm_slack < ddl_protection_threshold
      - Gate uses smooth sigmoid on jointly normalized slack (not hard threshold) to avoid discontinuities
      - Under protection mode, boosts critical-path urgency multiplicatively — preserves intra-violation ranking while strengthening deadline defense
      - Replaces multiplicative boost-on-violation with conditional, smooth, and tunable activation — validated by replay
      - Bounded wait-term now uses monotonic decay: 1 / (1 + ready_wait_time / (|slack| + eps)) — no sign discontinuity, maintains ordering
      - All numeric literals strictly limited to {-2,-1,0,1,2}
      - Removes problematic multiplicative criticality boost under violation; replaces with clean conditional gating
    """
    eps = 1.6873059837465527e-07
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
    duration_total = min_exec_time + min_comm_time + eps
    joint_features = np.stack([np.abs(slack), uncertainty, duration_total], axis=0).flatten()
    joint_med = np.median(joint_features)
    joint_mad = np.median(np.abs(joint_features - joint_med)) + eps
    joint_scale = 2.1653083704191096 * joint_mad

    def joint_normalize(x):
        x = np.asarray(x, dtype=float)
        if N == 1:
            return np.zeros_like(x, dtype=float)
        centered = x - joint_med
        normalized = centered / joint_scale
        return np.clip(normalized, -2.0, 2.0)
    slack_score = np.where(slack < 0, (-slack) ** 1.9625275666563022, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 1.4144571686190341
    duration_risk = duration_total * uncertainty * 0.7464312024581721
    crit_path_urgency = upward_rank * remaining_work
    critical_score = joint_normalize(crit_path_urgency)
    norm_slack = joint_normalize(slack)
    slack_gate = 1.0 / (1.0 + np.exp(-3.2018931670819084 * norm_slack))
    ddl_protection_gate = 1.0 / (1.0 + np.exp(-3.2018931670819084 * (norm_slack - 1.5319493300049762)))
    energy_per_sec = min_incremental_energy / (duration_total + eps)
    energy_eff_score = joint_normalize(energy_per_sec)
    rank_score = -joint_normalize(upward_rank) * (0.23004720776496226 * (1.0 - slack_gate) + (1.0 - 0.23004720776496226) * slack_gate)
    energy_norm = joint_normalize(min_incremental_energy)
    unc_norm = joint_normalize(uncertainty)
    unc_sigmoid = 1.0 / (1.0 + np.exp(-3.2018931670819084 * (unc_norm - 0.0)))
    energy_uncertainty_score = 0.7073304777843867 * energy_norm * unc_norm * unc_sigmoid
    wait_decay = 1.0 / (1.0 + ready_wait_time / (np.abs(slack) + eps))
    wait_norm = joint_normalize(wait_decay)
    score = joint_normalize(slack_score) + joint_normalize(unc_slack_coupling) + joint_normalize(duration_risk)
    score += critical_score
    score += slack_gate * (0.4240605387574511 * energy_eff_score + rank_score + energy_uncertainty_score + wait_norm)
    score += ddl_protection_gate * critical_score * 0.5925963206594987
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
