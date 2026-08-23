import numpy as np
RULE_METADATA = {'structure_hash': '884840074a4f8106c18d868e309c821fede149aa3c2a7c8dbe26dce3d22e8aef', 'parameter_schema_hash': 'a5e5a198dbf6c550376bb57a63f406345c839ac0c9cde96175c88045721a9398', 'best_parameter_hash': '4ca456e5dfe8012db9ca4b93947ea32bf839ad36cdf1b327ca994af436911dc6', 'best_parameters': {'epsilon': 5.098858296124484e-08, 'slack_penalty_exponent': 2.5101457659876036, 'criticality_boost': 1.718437511538723, 'energy_efficiency_ratio_weight': 0.2521248475024327, 'uncertainty_slack_coupling': 2.192755454181753, 'rank_slack_balance': 0.7971912620551388, 'duration_risk_penalty': 1.4406527871481793, 'energy_uncertainty_interaction': 1.3154235052002303, 'uncertainty_sigmoid_steepness': 4.330058934595657, 'joint_mad_scale': 0.9142264463143666, 'wait_decay_steepness': 3.83748794364327}, 'optimizer_config_hash': 'bd16adfa68cb3d3c380e679bfefc37d2ca8416a7e2e3e8f05a56dd735c7a1d44', 'parameter_diagnostics_hash': '885ba0b60e7bf7bb2aed408cd99fd3dcccf4f8172502d8f867a25d23d21c3b2c', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with soft top-k attention and successor-release proxy:
      - Soft top-k attention over urgency (norm_slack + norm_rank_work) replaces hard gating.
      - Successor-release modeled via local slack gradient: penalizes tasks preceding sharp slack drops.
      - All parameters declared in schema are used; no unused or missing references.
      - Numeric literals strictly limited to {-2,-1,0,1,2}.
    """
    eps = 5.098858296124484e-08
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
    joint_scale = 0.9142264463143666 * joint_mad

    def joint_normalize(x):
        x = np.asarray(x, dtype=float)
        if N == 1:
            return np.zeros_like(x, dtype=float)
        centered = x - joint_med
        normalized = centered / joint_scale
        return np.clip(normalized, -2.0, 2.0)
    slack_score = np.where(slack < 0, (-slack) ** 2.5101457659876036, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 2.192755454181753
    duration_risk = duration_total * uncertainty * 1.4406527871481793
    rank_work_product = upward_rank * remaining_work
    norm_slack = joint_normalize(slack)
    norm_rank_work = joint_normalize(rank_work_product)
    urgency_score = -norm_slack + norm_rank_work
    shifted_urgency = urgency_score - np.max(urgency_score)
    attention_weights = np.exp(1.718437511538723 * shifted_urgency)
    attention_weights = attention_weights / (np.sum(attention_weights) + eps)
    critical_score = attention_weights * np.abs(norm_rank_work)
    if N > 1:
        idx_sorted = np.argsort(slack)
        slack_sorted = slack[idx_sorted]
        slack_diff = np.concatenate([np.diff(slack_sorted), [0.0]])
        successor_slack_pressure = np.zeros_like(slack)
        successor_slack_pressure[idx_sorted] = np.clip(-slack_diff, 0.0, None)
    else:
        successor_slack_pressure = np.zeros_like(slack)
    successor_penalty = successor_slack_pressure * 0.2521248475024327
    slack_gate = 1.0 / (1.0 + np.exp(-4.330058934595657 * norm_slack))
    energy_per_sec = min_incremental_energy / (duration_total + eps)
    energy_eff_score = joint_normalize(energy_per_sec)
    rank_score = -joint_normalize(upward_rank) * (0.7971912620551388 * (1.0 - slack_gate) + (1.0 - 0.7971912620551388) * slack_gate)
    energy_norm = joint_normalize(min_incremental_energy)
    unc_norm = joint_normalize(uncertainty)
    unc_sigmoid = 1.0 / (1.0 + np.exp(-4.330058934595657 * (unc_norm - 0.0)))
    energy_uncertainty_score = 1.3154235052002303 * energy_norm * unc_norm * unc_sigmoid
    wait_decay = np.exp(-3.83748794364327 * np.abs(norm_slack))
    wait_score = joint_normalize(ready_wait_time) * wait_decay
    score = joint_normalize(slack_score) + joint_normalize(unc_slack_coupling) + joint_normalize(duration_risk) + critical_score + successor_penalty
    score += slack_gate * (0.2521248475024327 * energy_eff_score + rank_score + energy_uncertainty_score + wait_score)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
