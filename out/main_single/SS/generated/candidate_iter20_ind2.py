import numpy as np
RULE_METADATA = {'structure_hash': 'ef91977740aa7c1d5b3cf8ba7c4d41cfdd27fee5bc3315220ee3e42bd0ee37c3', 'parameter_schema_hash': 'd0ba37ee305ce0ee497e5f1f7876e1c3f7de4f46ad5c56e2a62ccd778464406d', 'best_parameter_hash': '0997f3b014b6301be0b866983bcf62608e0397d2d4134899bc2a9581b5442215', 'best_parameters': {'epsilon': 0.009464295529102376, 'criticality_scale': 0.7448757896182723, 'energy_sensitivity': 0.2086864568916733, 'energy_uncertainty_interaction': 0.6672435928519667, 'remaining_work_weight': 0.4859185522295867, 'uncertainty_gate_threshold': 0.27394541858722754, 'wait_decay': 0.4441907122447719, 'slack_penalty_linear_coeff': 2.043309238309802, 'duration_robustness_factor': 0.18303093546867416, 'slack_energy_suppression_strength': 0.6880610122037385, 'successor_release_steepness': 8.581273268817272}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': '07d67634088212d85fe2d422f5e99220bfdae34737df769c21192b6614b446ae', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule: merges Parent 2's hard feasibility gating and tunable energy suppression with Parent 1's successor-release gate.
       Structural novelty: replaces comm-energy coupling (removed to meet parameter limit) with *tighter release-gap activation* — uses sigmoidal gate on (slack - release_horizon) with explicit thresholding to sharpen bottleneck detection.
       Retains clipped-linear starvation relief, robustified duration penalty, and dual-gated energy suppression. All numeric literals are in [-2,2] or use np.finfo."""
    eps = 0.009464295529102376
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def robust_normalize(x):
        x = np.asarray(x, dtype=float)
        if N == 1:
            med = x[0]
            mad = eps
        else:
            med = np.median(x)
            mad = np.median(np.abs(x - med))
        spread = mad + eps
        return (x - med) / spread
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    hard_feasibility_gate = np.where(slack < 0.0, 0.0, 1.0)
    ddl_pressure = np.clip(-norm_slack, 0.0, 1.0)
    raw_slack_penalty = np.maximum(-slack, 0.0)
    norm_slack_penalty = robust_normalize(raw_slack_penalty)
    slack_penalty = 2.043309238309802 * norm_slack_penalty
    release_horizon = np.where(upward_rank > eps, remaining_work / upward_rank, np.finfo(float).max)
    release_gap = slack - release_horizon
    successor_release_gate = 1.0 / (1.0 + np.exp(-8.581273268817272 * np.clip(release_gap, -2.0, 2.0)))
    coupled_rank = norm_rank * (1.0 + 0.7448757896182723 * ddl_pressure)
    uncert_gate = np.where(norm_uncert > 0.27394541858722754, 1.0, 0.0)
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * hard_feasibility_gate
    wait_benefit = np.clip(0.4441907122447719 * ready_wait_time, 0.0, 2.0)
    duration_penalty = 0.18303093546867416 * norm_duration * ddl_pressure * hard_feasibility_gate
    slack_energy_suppress = np.where(slack < 0.0, 1.0 - 0.6880610122037385, 1.0)
    suppressed_norm_energy = norm_energy * slack_energy_suppress
    score = +np.clip(slack_penalty, -2.0, 2.0) - np.clip(coupled_rank, -2.0, 2.0) - np.clip(successor_release_gate, -2.0, 2.0) - 0.2086864568916733 * np.clip(suppressed_norm_energy, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip(duration_penalty, -2.0, 2.0) + 0.6672435928519667 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 0.4859185522295867 * np.clip(norm_work, -2.0, 2.0)
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.reshape(-1)
