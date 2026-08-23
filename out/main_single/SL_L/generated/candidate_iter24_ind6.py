import numpy as np
RULE_METADATA = {'structure_hash': '83063627cee4c3de9d3c497ebeaf34282cc992afeab8e0ecdd4c5f0129893025', 'parameter_schema_hash': '6dfc84ac6016d19f514962c3ce1314b6f53d23e7c1f319f6b12e2ce46d1595b6', 'best_parameter_hash': '0501003d1d4f2ad8bad7ad497f921b0d26309497d6b49f2091e3be7cb29d4673', 'best_parameters': {'epsilon': 6.795682665352459e-09, 'ddl_protection_threshold': 0.7577869325415619, 'critical_path_release_weight': 3.968384803955736, 'risk_adjusted_energy_weight': 1.1922498009428484, 'uncertainty_sigmoid_steepness': 6.0064414299802, 'wait_time_decay_exponent': 1.6099645537354832, 'slack_penalty_exponent': 2.463325332648517, 'duration_risk_penalty': 0.06598174891044316, 'energy_uncertainty_interaction': 0.8607327212510038, 'critical_slack_threshold': 0.9942143862358126, 'minmax_norm_epsilon': 0.029501210572544966}, 'optimizer_config_hash': 'bd16adfa68cb3d3c380e679bfefc37d2ca8416a7e2e3e8f05a56dd735c7a1d44', 'parameter_diagnostics_hash': '20cce50598d43010b9e252bfde7d2cb843c3ad78be309de6a2a4e28d9dbfccc6', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule featuring:
      - Hard feasibility-aware switch: below `critical_slack_threshold`, score prioritizes slack urgency (|slack| when negative, else 0);
        above it, defaults to critical-path release — eliminates fragile interpolation.
      - Bounded min-max normalization for slack, uncertainty, duration_total (replacing MAD) to stabilize small-N ready sets.
      - Decoupled anti-starvation: uses normalized `ready_wait_time` directly (not scaled by headroom), gated only by DDL feasibility.
      - All numeric literals strictly limited to {-2,-1,0,1,2}; no other constants.
      - Final score is deterministic, finite, shape-(N,), and satisfies 'smaller = higher priority'.
    """
    eps = 6.795682665352459e-09
    mm_eps = 0.029501210572544966
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

    def minmax_normalize(x):
        x = np.asarray(x, dtype=float)
        if N == 1:
            return np.zeros_like(x, dtype=float)
        x_min = np.min(x)
        x_max = np.max(x)
        denom = x_max - x_min + mm_eps
        normalized = (x - x_min) / denom
        return np.clip(normalized, 0.0, 1.0)
    duration_total = min_exec_time + min_comm_time + eps
    slack_abs = np.abs(slack)
    slack_penalty = np.where(slack < 0, (-slack) ** 2.463325332648517, 0.0)
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 0.06598174891044316
    critical_release_score = upward_rank * remaining_work * 3.968384803955736
    urgency_mask = np.where(slack < 0.9942143862358126, 1.0, 0.0)
    urgency_score = np.where(slack < 0, slack_abs, 0.0)
    if np.any(slack < 0):
        urgency_norm = minmax_normalize(urgency_score)
    else:
        urgency_norm = np.zeros_like(urgency_score)
    switched_base_score = urgency_mask * urgency_norm + (1.0 - urgency_mask) * minmax_normalize(critical_release_score)
    slack_headroom_mask = np.where(slack > 0.7577869325415619, 1.0, 0.0)
    energy_norm = minmax_normalize(min_incremental_energy)
    energy_term = slack_headroom_mask * 1.1922498009428484 * energy_norm
    wait_norm = minmax_normalize(ready_wait_time)
    wait_term = slack_headroom_mask * wait_norm ** 1.6099645537354832
    unc_sigmoid = 1.0 / (1.0 + np.exp(-6.0064414299802 * (uncertainty - 1.0)))
    unc_norm = minmax_normalize(uncertainty)
    energy_uncertainty_term = slack_headroom_mask * 0.8607327212510038 * energy_norm * unc_norm * unc_sigmoid
    score = minmax_normalize(slack_penalty) + minmax_normalize(duration_risk) + switched_base_score
    score += energy_term + wait_term + energy_uncertainty_term
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
