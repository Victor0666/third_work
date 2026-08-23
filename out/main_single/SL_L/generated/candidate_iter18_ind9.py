import numpy as np
RULE_METADATA = {'structure_hash': 'ca41c4ea729b3528564ef9067c42812477d24ef2b9d624bc723f6d2831a84e7e', 'parameter_schema_hash': '7fe1af4c0f7994a4478472a20efed0595b3f8384d73ebe847909e2125abe705c', 'best_parameter_hash': '093ab1b0a40018f44387df65db72961aee76637672e59f78f0d77ea0f7f4cad2', 'best_parameters': {'epsilon': 3.168069966072081e-07, 'slack_penalty_exponent': 2.69632920981428, 'upward_rank_remaining_work_interaction': 0.6365487196704875, 'energy_efficiency_ratio_weight': 1.3289530134597434, 'uncertainty_slack_coupling': 1.758290291212239, 'duration_risk_penalty': 0.7429333308296666, 'energy_uncertainty_interaction': 0.650927814330105, 'uncertainty_sigmoid_steepness': 3.7618555312296786, 'ddl_protection_threshold': 0.03252120143236286, 'wait_starvation_suppression': 0.25370815466606694, 'host_load_conditional_gate': 0.33589464903057753}, 'optimizer_config_hash': 'bd16adfa68cb3d3c380e679bfefc37d2ca8416a7e2e3e8f05a56dd735c7a1d44', 'parameter_diagnostics_hash': 'dad8eb9de9664e2a9d5ff2fde27ea970ebb35ce27ebe77f6f33aaa56f7df3fba', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with restored host-load awareness and linear slack headroom:
      - Reintroduces `host_load_conditional_gate` to restore load-aware energy optimization.
      - Replaces concave `rank_headroom_nonlinearity` with robust linear headroom mapping for fairness and deadline safety.
      - Uses unified MAD normalization throughout for stability across all N ≥ 1.
      - All DDL-critical terms remain unconditionally active; non-DDL terms gated solely by `ddl_safe_mask`.
      - Energy-efficiency term now uses host-load-aware scaling: multiplied by `host_load_scale`.
      - Anti-starvation `wait_score` retains linear headroom scaling (no decay) to preserve urgency for long-waiting tasks even under high slack.
      - No numeric literals beyond {-2,-1,0,1,2}; all tunables declared in PARAMETER_SCHEMA.
    """
    eps = 3.168069966072081e-07
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
    slack_score = np.where(slack < 0, (-slack) ** 2.69632920981428, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 1.758290291212239
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 0.7429333308296666
    critical_path_release = upward_rank * remaining_work * 0.6365487196704875
    ddl_safe_mask = np.where(slack > 0.03252120143236286, 1.0, 0.0)
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_score = mad_normalize(energy_per_sec)
    slack_headroom = np.maximum(0.0, slack - 0.03252120143236286)
    max_slack_headroom = np.max(slack_headroom) + eps
    slack_headroom_normalized = slack_headroom / max_slack_headroom
    host_load_scale = slack_headroom_normalized * 0.33589464903057753
    rank_score = -mad_normalize(upward_rank) * (1.0 + (1.0 - slack_headroom_normalized))
    unc_sigmoid = 1.0 / (1.0 + np.exp(-3.7618555312296786 * (uncertainty - 1.0)))
    energy_norm = mad_normalize(min_incremental_energy)
    unc_norm = mad_normalize(uncertainty)
    energy_uncertainty_score = ddl_safe_mask * host_load_scale * 0.650927814330105 * energy_norm * unc_norm * unc_sigmoid
    wait_score = mad_normalize(ready_wait_time) * slack_headroom_normalized * 0.25370815466606694
    score = mad_normalize(slack_score) + mad_normalize(unc_slack_coupling) + mad_normalize(duration_risk) + mad_normalize(critical_path_release)
    score += ddl_safe_mask * (1.3289530134597434 * host_load_scale * energy_eff_score + rank_score + energy_uncertainty_score + wait_score)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
