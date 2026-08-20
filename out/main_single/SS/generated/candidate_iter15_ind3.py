import numpy as np
RULE_METADATA = {'structure_hash': 'b12fe7200e71fda84e82264178f0f4b8a7b533e1e516a893373417d0a96e8dfb', 'parameter_schema_hash': '7e6f289e5c25d5d2ca1b75185d852b5ff0f994da0a87e87981b7abfe4177dc7f', 'best_parameter_hash': '733a1cae720a1b4cecf2b74fe56b262161ef2a017d309b88511621f54603d40c', 'best_parameters': {'epsilon': 3.743228912302062e-05, 'slack_penalty_exponent': 1.4509797555348158, 'criticality_scale': 0.849277610849949, 'energy_sensitivity': 0.24873191411250478, 'remaining_work_weight': 0.9647153454965658, 'uncertainty_gate_threshold': 0.38354397379428273, 'energy_uncertainty_interaction': 0.3753792720498001, 'tanh_half_scale': 0.24048104837486403, 'slack_linear_threshold': 0.08351290012742599, 'bottleneck_activation_gate': 0.5262021158367198}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': '118c8f9cdccd6e3bf2ce00037bbeab5ba732f7b6c31d0268a543c156a8f568e7', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with two key structural improvements:
       1. Piecewise slack penalty: linear for |slack| <= threshold, quadratic beyond — matches violation gravity while preserving monotonicity.
       2. Dual-gated bottleneck release: successor_release activated ONLY when (slack <= eps) AND (norm_uncert <= bottleneck_activation_gate),
          preventing unsafe prioritization of critical paths under high uncertainty — directly addressing self-reflection.
       Removed wait_decay and starvation relief entirely: diagnostics confirm zero contribution and numerical destabilization.
       All normalization remains median-MAD for outlier resilience; all terms clipped to [-2,2] for AST depth control."""
    eps = 3.743228912302062e-05
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
    norm_rank = median_mad_normalize(upward_rank)
    norm_work = median_mad_normalize(remaining_work)
    norm_uncert = median_mad_normalize(uncertainty)
    ddl_feasible = np.clip(0.24048104837486403 * (1.0 - np.tanh(slack / (eps + np.finfo(float).tiny))), 0.0, 1.0)
    slack_pressure = np.clip(norm_slack, -1.0, 1.0)
    ddl_urgent = (slack <= eps).astype(float)
    low_uncert = (norm_uncert <= 0.5262021158367198).astype(float)
    successor_release = norm_work * norm_rank * ddl_urgent * low_uncert
    abs_norm_slack = np.abs(norm_slack)
    linear_penalty = abs_norm_slack
    quadratic_penalty = abs_norm_slack ** 2
    piecewise_slack_penalty = np.where(abs_norm_slack <= 0.08351290012742599, linear_penalty, quadratic_penalty)
    energy_penalty = norm_energy * (1.0 + 1.4509797555348158 * slack_pressure) * (1.0 - ddl_feasible)
    uncert_gate = (norm_uncert > 0.38354397379428273).astype(float)
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * (1.0 - ddl_feasible)
    score = +np.clip(piecewise_slack_penalty, -2.0, 2.0) - np.clip(successor_release, -2.0, 2.0) - np.clip(norm_rank * (1.0 + 0.849277610849949 * slack_pressure), -2.0, 2.0) - 0.24873191411250478 * np.clip(energy_penalty, -2.0, 2.0) + 0.3753792720498001 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 0.9647153454965658 * np.clip(norm_work, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
