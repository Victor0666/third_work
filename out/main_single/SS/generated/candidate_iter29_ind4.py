import numpy as np
RULE_METADATA = {'structure_hash': 'b754d4efc32b050788988465de2b2a450a2feaca23d90daeadcc261ee93951b5', 'parameter_schema_hash': 'fb337780f36ce7952fc96327b02b7f7fdf108400d3d6cefd9d3ddaf688592b91', 'best_parameter_hash': '758bc08ae2daf4b2737e50b90a1ff6234a5f51184b8f152045c5998efb69d931', 'best_parameters': {'epsilon': 0.003928534977645774, 'criticality_scale': 1.9286614351501659, 'energy_sensitivity': 1.0833346647883206, 'energy_uncertainty_interaction': 0.9999447925251983, 'remaining_work_weight': 0.803638808190504, 'uncertainty_gate_threshold': 0.15230860558997944, 'wait_decay': 0.6039218936212614, 'ddl_pressure_gate_threshold': 0.8253330626185391, 'slack_penalty_exponent': 2.5382548474622224, 'host_load_steepness': 5.538572950505868, 'host_load_activation_offset': 0.15544644256745127}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': '4fdf56f1e4b47e1a647a0c93f6a5455d570809ed46c71603b81915867227dfbd', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule: merges Parent 2's robust ddl_pressure and successor-release with Parent 1's slack_penalty_exponent;
       replaces static host_load_surrogate with sigmoid-gated duration-uncertainty coupling for sharper congestion response;
       introduces slack-aware anti-starvation: wait_benefit scaled by (1 - ddl_pressure) AND activated only when slack > 0;
       uses clipped MAD-normalized slack_penalty as primary urgency signal, enhanced by exponent for asymmetric lateness risk;
       eliminates redundant rank_slack_coupling and rank_gate — replaced by unified criticality_scale × ddl_pressure;
       all intermediate terms clipped to [-2,2] to ensure bounded AST depth and stability."""
    eps = 0.003928534977645774
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def median_mad_normalize(x):
        x = np.copy(x)
        if N == 1:
            med = x[0]
            mad = eps
        else:
            med = np.median(x)
            mad = np.median(np.abs(x - med))
        spread = mad if mad > eps else eps
        return (x - med) / spread
    norm_slack = median_mad_normalize(slack)
    norm_energy = median_mad_normalize(min_incremental_energy)
    norm_duration = median_mad_normalize(min_exec_time + min_comm_time)
    norm_rank = median_mad_normalize(upward_rank)
    norm_work = median_mad_normalize(remaining_work)
    norm_wait = median_mad_normalize(ready_wait_time)
    norm_uncert = median_mad_normalize(uncertainty)
    ddl_pressure = 1.0 / (1.0 + np.abs(slack) + eps)
    ddl_breach = (slack <= 0.0).astype(float)
    critical_path_urgency = norm_rank * norm_work * ddl_breach
    successor_release = min_exec_time * norm_rank * ddl_breach
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 2.5382548474622224
    norm_slack_penalty = np.clip(median_mad_normalize(raw_slack_penalty), -2.0, 2.0)
    load_activation = norm_duration * norm_uncert
    host_load_gate = 1.0 / (1.0 + np.exp(-5.538572950505868 * (load_activation - 0.15544644256745127)))
    host_load_surrogate = load_activation * host_load_gate * ddl_pressure
    energy_uncert_gate = (ddl_pressure > 0.8253330626185391).astype(float) * (norm_uncert >= 0.15230860558997944).astype(float)
    energy_uncert_penalty = norm_energy * norm_uncert * energy_uncert_gate
    wait_eligible = (slack > 0.0).astype(float)
    wait_benefit = np.exp(-0.6039218936212614 * ready_wait_time) * (1.0 - ddl_pressure) * wait_eligible
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(critical_path_urgency, -2.0, 2.0) - np.clip(successor_release, -2.0, 2.0) - 1.0833346647883206 * np.clip(norm_energy * ddl_pressure, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip(host_load_surrogate, -2.0, 2.0) + 0.9999447925251983 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 0.803638808190504 * np.clip(norm_work, -2.0, 2.0) + 1.9286614351501659 * np.clip(norm_rank * ddl_pressure, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
