import numpy as np
RULE_METADATA = {'structure_hash': 'aad5bc013096b9c4ea2cf1bf11231807dd69446c238f908b6d6ae7ac8652808e', 'parameter_schema_hash': 'a855aee71f1a9f90c5e352cd4a8d20a55a7ba6af0f32f52c30eadec9860d746f', 'best_parameter_hash': 'a1757acc52097173376bb61deab93e6cde168b8ff643eb8798871eea6a4a335b', 'best_parameters': {'epsilon': 7.79864852274516e-08, 'slack_penalty_exponent': 2.3395016617078133, 'upward_rank_remaining_work_interaction': 0.41120918372469106, 'energy_efficiency_ratio_weight': 1.2698122914052912, 'uncertainty_slack_coupling': 1.9396770275242472, 'duration_risk_penalty': 0.7694191170288369, 'energy_uncertainty_interaction': 0.5831540549239675, 'uncertainty_sigmoid_steepness': 1.6579243437842717, 'ddl_protection_threshold': 0.921974900885862, 'wait_starvation_suppression': 0.48879638393872504, 'rank_headroom_nonlinearity': 0.30285988600842817}, 'optimizer_config_hash': 'bd16adfa68cb3d3c380e679bfefc37d2ca8416a7e2e3e8f05a56dd735c7a1d44', 'parameter_diagnostics_hash': 'f88683ba7488550edb127bacb1328808be2645f3c203db4379fc6f186bfda7ec', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule with all declared parameters used and no unused entries:
      - Removed `host_load_conditional_gate` (previously unused) and simplified gating to single `ddl_safe_mask`.
      - Retains concave `rank_headroom_nonlinearity`, unconditional critical-path term, and robust MAD normalization.
      - All DDL-critical terms are unconditionally active; non-DDL terms are gated *only* by `ddl_safe_mask`.
      - `wait_starvation_suppression` is applied directly — no host-load scaling, eliminating the need for the removed parameter.
      - Uses only allowed numeric literals (-2,-1,0,1,2); all tunables are in PARAMETER_SCHEMA.
      - Final score is finite, shape-(N,), deterministic, and satisfies all interface contracts.
    """
    eps = 7.79864852274516e-08
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
    slack_score = np.where(slack < 0, (-slack) ** 2.3395016617078133, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 1.9396770275242472
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 0.7694191170288369
    critical_path_release = upward_rank * remaining_work * 0.41120918372469106
    ddl_safe_mask = np.where(slack > 0.921974900885862, 1.0, 0.0)
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_score = mad_normalize(energy_per_sec)
    slack_headroom = np.maximum(0.0, slack - 0.921974900885862)
    max_slack_headroom = np.max(slack_headroom) + eps
    slack_headroom_normalized = slack_headroom / max_slack_headroom
    slack_headroom_decay = slack_headroom_normalized ** 0.30285988600842817
    rank_score = -mad_normalize(upward_rank) * (1.0 + (1.0 - slack_headroom_decay))
    unc_sigmoid = 1.0 / (1.0 + np.exp(-1.6579243437842717 * (uncertainty - 1.0)))
    energy_norm = mad_normalize(min_incremental_energy)
    unc_norm = mad_normalize(uncertainty)
    energy_uncertainty_score = ddl_safe_mask * 0.5831540549239675 * energy_norm * unc_norm * unc_sigmoid
    wait_score = mad_normalize(ready_wait_time) * slack_headroom_decay * 0.48879638393872504
    score = mad_normalize(slack_score) + mad_normalize(unc_slack_coupling) + mad_normalize(duration_risk) + mad_normalize(critical_path_release)
    score += ddl_safe_mask * (1.2698122914052912 * energy_eff_score + rank_score + energy_uncertainty_score + wait_score)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
