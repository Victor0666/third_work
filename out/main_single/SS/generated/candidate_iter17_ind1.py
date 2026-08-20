import numpy as np
RULE_METADATA = {'structure_hash': '406a2783978ad23cd98a572e26fffffe9829617792530c0baa7a7da5d43925bc', 'parameter_schema_hash': 'd2d8babee680dc20704524bfa8f97cf9bc14e444c4eb288983c8e9efa5189db9', 'best_parameter_hash': '25513cfbcd22af145aaa74730b4d9061e34de304143d315f648fd55a4d12e810', 'best_parameters': {'epsilon': 0.00012476039493949657, 'criticality_scale': 2.047419254010298, 'energy_sensitivity': 0.5571279535323509, 'energy_uncertainty_interaction': 0.18015689195191048, 'remaining_work_weight': 0.6068272778292467, 'ddl_protection_gate_slope': 6.976816043461547, 'uncertainty_gate_threshold': 0.39537624648515235, 'wait_decay': 0.579232547242282, 'slack_penalty_linear_coeff': 0.5350854126046283, 'duration_robustness_factor': 0.7011865106467883, 'starvation_saturation_offset': 2.414197712884674e-08, 'rank_uncert_coupling_strength': 0.16309195072260513}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': '21901ead0816332144efb8e14cc6aa85518908130cee83bfc0a5b7d29ca4f6dd', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule: merges best practices from both parents.
       - Retains Parent 2's robust median-MAD normalization, linear slack penalty, unified ddl_gate, and duration_robustness.
       - Adopts Parent 1's crisp *dual-gated* rank-uncertainty coupling but replaces binary ddl_infeasible with graded ddl_pressure (smooth monotonic activation).
       - Introduces novel starvation relief using clipped *logarithmic* form with saturating offset — preserves discriminability at low wait times while preventing singularity.
       - Removes redundant terms (e.g., separate exec_penalty) and unifies all feasibility-sensitive terms under ddl_gate + ddl_pressure for tighter DDL alignment.
       - All interactions bounded and clipped to [-2,2] for stability and ranking integrity."""
    eps = 0.00012476039493949657
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
    ddl_gate = np.where(slack < 0.0, 0.0, np.clip(slack, 0.0, 1.0))
    ddl_pressure = np.clip(-norm_slack, 0.0, 1.0)
    raw_slack_penalty = np.maximum(-slack, 0.0)
    norm_slack_penalty = robust_normalize(raw_slack_penalty)
    slack_penalty = 0.5350854126046283 * norm_slack_penalty
    successor_release = norm_rank * norm_work * ddl_pressure
    coupled_rank = norm_rank * (1.0 + 2.047419254010298 * ddl_pressure)
    uncert_gate = np.where(norm_uncert > 0.39537624648515235, 1.0, 0.0)
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_gate
    wait_benefit = np.log1p(0.579232547242282 * (ready_wait_time + 2.414197712884674e-08))
    wait_benefit = np.clip(wait_benefit, 0.0, 2.0)
    duration_penalty = 0.7011865106467883 * norm_duration * ddl_pressure * ddl_gate
    rank_uncert_coupling = norm_rank * norm_uncert * ddl_pressure * 0.16309195072260513
    score = +np.clip(slack_penalty, -2.0, 2.0) - np.clip(successor_release, -2.0, 2.0) - np.clip(coupled_rank, -2.0, 2.0) - 0.5571279535323509 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip(duration_penalty, -2.0, 2.0) + np.clip(rank_uncert_coupling, -2.0, 2.0) + 0.18015689195191048 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 0.6068272778292467 * np.clip(norm_work, -2.0, 2.0) + np.clip(ddl_pressure * 6.976816043461547, -2.0, 2.0)
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.reshape(-1)
