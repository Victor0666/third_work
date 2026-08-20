import numpy as np
RULE_METADATA = {'structure_hash': 'b4dece1d7193946054eaa535a5096f17a0eab012d9dd443be3dedab8c5b03ab5', 'parameter_schema_hash': '22f9bedc361c55d78ca432d99e90044e78a6eb7e23ef159d9ac1882d3e9c1535', 'best_parameter_hash': 'd261c8e5363ccf70a3242f34bb59785b80f860f9e4954ac05b0ec7dab2a02e51', 'best_parameters': {'epsilon': 0.0007879988244281781, 'slack_risk_penalty': 8.08039118414321, 'slack_urgency_gain': 4.037859267517651, 'energy_efficiency_weight': 0.19735255581484684, 'criticality_weight': 2.375352219269204, 'duration_uncertainty_ratio': 0.4599876122109327, 'wait_decay_rate': 0.21387360591838647, 'uncertainty_slack_interaction': 2.7148476037185776, 'finfo_max_scale': 1272398.7970615502, 'ddl_protection_gate_width': 0.18492784123349948, 'load_successor_release_weight': 1.0629270459386}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': '8d81915c959c277e524511d63fabf2b2fdf28cbb887457a2eecb16d4e66d4a34', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule with smooth DDL protection gate and successor-release interaction.
    
    Key structural mutations:
      - Replaces hard `slack_norm >= 0` gating with *smooth sigmoid gate* centered at slack=0,
        controlled by `ddl_protection_gate_width` — eliminates discontinuities while preserving
        strong deadline-driven activation near zero slack.
      - Adds `load_successor_release_weight`-scaled term: computes normalized product of
        `upward_rank` and `remaining_work`, then applies smooth gate — promotes tasks that
        release large downstream critical work early under deadline pressure.
      - Retains robust normalization, exponential wait saturation, and bounded linear duration-uncertainty blend.
      - All gates are monotonic and differentiable; no conditionals on raw magnitudes.
      - Uses `np.tanh` for numerically stable, bounded smooth step instead of fragile piecewise logic.
    """
    eps = 0.0007879988244281781
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    rem_work = np.asarray(remaining_work, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)

    def normalize(x):
        x = np.asarray(x)
        abs_x = np.abs(x)
        scale = np.mean(abs_x) if np.all(np.isfinite(abs_x)) and np.mean(abs_x) > eps else eps
        return x / (scale + eps)
    gate_center = 0.0
    gate_width = 0.18492784123349948 + eps
    slack_gate = np.tanh((slk - gate_center) / gate_width)
    slack_penalty = 8.08039118414321 * np.where(slk < 0, np.abs(slk), 0.0) - 4.037859267517651 * np.where(slk > 0, slk, 0.0)
    slack_penalty = normalize(slack_penalty + eps)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.19735255581484684 * normalize(inv_energy)
    rank_score = -2.375352219269204 * normalize(rank + eps) * (1.0 + slack_gate) / 2.0
    successor_release_potential = rank * rem_work
    successor_release_score = -1.0629270459386 * normalize(successor_release_potential + eps) * (1.0 - slack_gate) / 2.0
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.4599876122109327 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_sat = 1.0 - np.exp(-0.21387360591838647 * wait)
    wait_score = -normalize(wait_sat + eps)
    unc_slack_interaction = 2.7148476037185776 * uncert_norm * np.maximum(0.0, -slack_gate)
    score = slack_penalty + energy_score + rank_score + successor_release_score + dur_score + wait_score + unc_slack_interaction
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 1272398.7970615502
    min_safe = -finfo.max / 1272398.7970615502
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
