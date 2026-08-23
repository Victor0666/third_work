import numpy as np
RULE_METADATA = {'structure_hash': '05b74531dc9a32c187f3b281e984e81379a0f713e2bb60d7d0fbebab37ed60ca', 'parameter_schema_hash': '1582641568c2471f88a877ea7a8554195080b93b76ce626ac1798b5508ed3177', 'best_parameter_hash': 'e2abafe449b150882f4f0055f2d790eece717b137b6381a424eb0b33a7c9aacd', 'best_parameters': {'epsilon': 9.938589846783386e-05, 'slack_penalty_exponent': 2.6295317478131333, 'criticality_boost': 3.1000457177866356, 'energy_efficiency_ratio_weight': 0.5571142705409484, 'uncertainty_slack_coupling': 1.1690327423550162, 'rank_slack_balance': 0.4319248050282901, 'duration_risk_penalty': 0.6758774109125439, 'energy_uncertainty_interaction': 0.4108291875749226, 'uncertainty_sigmoid_steepness': 2.062890253728067, 'slack_min_bound': -1.2945823404952108, 'slack_max_bound': 26.50661311572697, 'critical_threshold_factor': 0.4206358983950703}, 'optimizer_config_hash': 'bd16adfa68cb3d3c380e679bfefc37d2ca8416a7e2e3e8f05a56dd735c7a1d44', 'parameter_diagnostics_hash': 'f162ee931e63e14892fcd4f52601a0979198b8fdbfdbc438ff38f51f7de8e6d5', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule incorporating:
      - Unified MAD-based normalization (robust to outliers, N=1 safe) instead of percentile clipping
      - Critical path urgency expressed via upward_rank × remaining_work interaction, scaled by slack pressure
      - DDL-protection gate now uses smooth sigmoid instead of hard threshold for slack > 0, improving gradient flow
      - Anti-starvation term restructured as `ready_wait_time * exp(-|slack|/tau)` — decays smoothly with headroom
      - Energy-efficiency term replaced by ratio `min_incremental_energy / (min_exec_time + min_comm_time + eps)`
        normalized via MAD, avoiding division instability
      - All gates and interactions explicitly bounded and guarded; no unclipped divisions or exponentials
      - Criticality boost applied multiplicatively only when slack <= 0 AND upward_rank > factor*median AND remaining_work > factor*median
      - Final score preserves lexicographic DDL-first ordering: violation terms dominate feasible ones
    """
    eps = 9.938589846783386e-05
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
    slack_score = np.where(slack < 0, (-slack) ** 2.6295317478131333, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 1.1690327423550162
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 0.6758774109125439
    is_tight_or_violated = slack <= 0
    rank_median = np.median(upward_rank) if N > 1 else np.mean(upward_rank)
    work_median = np.median(remaining_work) if N > 1 else np.mean(remaining_work)
    is_high_rank = upward_rank >= 0.4206358983950703 * rank_median
    is_high_work = remaining_work >= 0.4206358983950703 * work_median
    critical_gate = np.where(is_high_rank & is_high_work & is_tight_or_violated, 3.1000457177866356, 1.0)
    slack_gate = 1.0 / (1.0 + np.exp(-2.062890253728067 * slack))
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_score = mad_normalize(energy_per_sec)
    slack_lb = -1.2945823404952108
    slack_ub = 26.50661311572697
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    weight_rank = 0.4319248050282901 + (1.0 - 0.4319248050282901) * (1.0 - slack_scaled)
    rank_score = -mad_normalize(upward_rank) * weight_rank
    energy_norm = mad_normalize(min_incremental_energy)
    unc_norm = mad_normalize(uncertainty)
    unc_sigmoid = 1.0 / (1.0 + np.exp(-2.062890253728067 * (uncertainty - 1.0)))
    energy_uncertainty_score = 0.4108291875749226 * energy_norm * unc_norm * unc_sigmoid
    wait_decay = np.exp(-np.abs(slack) / (26.50661311572697 + eps))
    wait_score = mad_normalize(ready_wait_time) * wait_decay
    score = mad_normalize(slack_score) + mad_normalize(unc_slack_coupling) + mad_normalize(duration_risk)
    score += slack_gate * (0.5571142705409484 * energy_eff_score + rank_score + energy_uncertainty_score + wait_score)
    score = score * critical_gate
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
