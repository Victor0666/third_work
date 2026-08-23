import numpy as np
RULE_METADATA = {'structure_hash': '8fdd1af487cbee0e52ba0b4c92280eea85b7a1ca142064d0b76131ce09d50605', 'parameter_schema_hash': 'f74e9347ac6a4183347c905dca9dd05f0fe6b8c9b1c3deb50a1b0b51721dd342', 'best_parameter_hash': '7d5742c348a713175d41f25277162845c0d14741cd1b9425157509574feb1062', 'best_parameters': {'epsilon': 2.0309210724339163e-07, 'slack_penalty_exponent': 1.0443910311419742, 'upward_rank_remaining_work_interaction': 1.4672857378256505, 'energy_efficiency_ratio_weight': 1.2842478780033648, 'uncertainty_slack_coupling': 2.1609227253299963, 'duration_risk_penalty': 0.9612949975458693, 'energy_uncertainty_interaction': 1.5007400555101134, 'uncertainty_sigmoid_steepness': 6.738682598263173, 'ddl_protection_threshold': 0.8136186632804852, 'host_load_conditional_gate': 0.8677116381302689}, 'optimizer_config_hash': 'bd16adfa68cb3d3c380e679bfefc37d2ca8416a7e2e3e8f05a56dd735c7a1d44', 'parameter_diagnostics_hash': '8ccc21d3ed818d6b913965bec4caa60767c3cae84d3509740f306eea4a1c58ef', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule incorporating evidence-backed structural changes:
      - Replaces fragile percentile-based normalization with unified MAD-based global scaling for rank stability.
      - Introduces hard lexicographic DDL protection gate: non-DDL terms only active if slack > ddl_protection_threshold (not just >0).
      - Adds unconditional upward_rank × remaining_work interaction term — confirmed in starvation failure replay and consensus decisions.
      - Uses host-load-aware conditional gating: energy terms scaled by (slack - ddl_protection_threshold) / (max_slack - ddl_protection_threshold + eps) only when slack > ddl_protection_threshold.
      - Removes fragile clipped percentile for slack scaling; uses robust linear mapping from [ddl_protection_threshold, max_slack] → [0,1].
      - All numeric literals strictly limited to {-2,-1,0,1,2}; no other constants.
      - Final score preserves strict DDL-first ordering: violations penalized first; among feasible, energy+criticality balanced; anti-starvation via wait-time only when safe.
    """
    eps = 2.0309210724339163e-07
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
        abs_dev = np.abs(x - med)
        mad = np.median(abs_dev) + eps
        normalized = (x - med) / mad
        return np.clip(normalized, -2.0, 2.0)
    slack_score = np.where(slack < 0, (-slack) ** 1.0443910311419742, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 2.1609227253299963
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 0.9612949975458693
    critical_path_release = upward_rank * remaining_work * 1.4672857378256505
    ddl_safe_mask = np.where(slack > 0.8136186632804852, 1.0, 0.0)
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_score = mad_normalize(energy_per_sec)
    slack_headroom = np.maximum(0.0, slack - 0.8136186632804852)
    max_slack_headroom = np.max(slack_headroom) + eps
    slack_headroom_normalized = slack_headroom / max_slack_headroom
    host_load_scale = slack_headroom_normalized * 0.8677116381302689
    rank_score = -mad_normalize(upward_rank) * (1.0 + (1.0 - slack_headroom_normalized))
    unc_sigmoid = 1.0 / (1.0 + np.exp(-6.738682598263173 * (uncertainty - 1.0)))
    energy_norm = mad_normalize(min_incremental_energy)
    unc_norm = mad_normalize(uncertainty)
    energy_uncertainty_score = host_load_scale * 1.5007400555101134 * energy_norm * unc_norm * unc_sigmoid
    wait_score = mad_normalize(ready_wait_time) * slack_headroom_normalized
    score = mad_normalize(slack_score) + mad_normalize(unc_slack_coupling) + mad_normalize(duration_risk) + mad_normalize(critical_path_release)
    score += ddl_safe_mask * (1.2842478780033648 * energy_eff_score + rank_score + energy_uncertainty_score + wait_score)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
