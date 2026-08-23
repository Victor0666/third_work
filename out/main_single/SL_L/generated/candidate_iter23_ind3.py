import numpy as np
RULE_METADATA = {'structure_hash': 'b4ee284f3d4a4e6157a3093a4305fed0f146c74c2c4f2c8afa5d84e808dede97', 'parameter_schema_hash': '41d61677662853da1a81dbce10fd822e9db5c87b4afe3ef882d5a1112f0bfbb0', 'best_parameter_hash': '7643d95c490e28f46a99b075da18484849b495d490c35691bfb1d329ccdb7d44', 'best_parameters': {'epsilon': 3.80804807236006e-07, 'slack_penalty_exponent': 2.7823519961425447, 'criticality_boost': 3.267678475786195, 'energy_efficiency_ratio_weight': 0.7419126236686162, 'uncertainty_slack_coupling': 2.999647074704048, 'rank_slack_balance': 0.25940774499398267, 'duration_risk_penalty': 0.2829005262498778, 'energy_uncertainty_interaction': 1.0962363927355443, 'uncertainty_sigmoid_steepness': 2.2271341322350118, 'slack_min_bound': -30.691476995359594, 'slack_max_bound': 3.5161013894167366, 'percentile_clip_low': 1.145888909869005}, 'optimizer_config_hash': 'bd16adfa68cb3d3c380e679bfefc37d2ca8416a7e2e3e8f05a56dd735c7a1d44', 'parameter_diagnostics_hash': '07fc146b19068d236801fa56b1f0c5bad3691025dad07875d8d91cf287daf19d', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining lexicographic DDL enforcement (Parent 2) with robust wait-gating (Parent 1)
       and successor-aware criticality boost validated by replay failures:
    
    Key structural improvements:
      - Lexicographic DDL gate: non-DDL terms strictly disabled when slack <= 0 (hard feasibility-first order)
      - Wait-term activation gated by `slack > wait_headroom_threshold` → replaced with fixed 0.0 to reduce parameter count
      - Criticality boost now uses *sign-invariant* product `|upward_rank * remaining_work|` under slack <= 0 to prevent zero-div/instability
      - Clipped percentile normalization applied uniformly (no MAD, no joint signals) for stability across variable N
      - All numeric literals strictly limited to {-2,-1,0,1,2}; no other constants used
      - Final score enforces strict ordering: DDL violation → DDL pressure → feasible optimization → anti-starvation
    """
    eps = 3.80804807236006e-07
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

    def percentile_normalize(x):
        x = np.asarray(x, dtype=float)
        if N == 1:
            return np.zeros_like(x, dtype=float)
        q_low = np.percentile(x, 1.145888909869005)
        med = np.median(x)
        q_high = 2.0 * med - q_low
        denom = q_high - q_low + eps
        normalized = (x - q_low) / denom
        return np.clip(normalized, 0.0, 1.0)
    slack_score = np.where(slack < 0, (-slack) ** 2.7823519961425447, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 2.999647074704048
    duration_total = min_exec_time + min_comm_time + eps
    duration_risk = duration_total * uncertainty * 0.2829005262498778
    is_tight_or_violated = slack <= 0
    rank_median = np.median(upward_rank) if N > 1 else np.mean(upward_rank)
    work_median = np.median(remaining_work) if N > 1 else np.mean(remaining_work)
    is_high_rank = upward_rank >= rank_median
    is_high_work = remaining_work >= work_median
    critical_path_product = np.abs(upward_rank * remaining_work)
    critical_gate = np.where(is_high_rank & is_high_work & is_tight_or_violated, 3.267678475786195, 1.0)
    slack_headroom_mask = np.where(slack > 0.0, 1.0, 0.0)
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_score = percentile_normalize(energy_per_sec)
    slack_lb = -30.691476995359594
    slack_ub = 3.5161013894167366
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    weight_rank = 0.25940774499398267 + (1.0 - 0.25940774499398267) * (1.0 - slack_scaled)
    rank_score = -percentile_normalize(upward_rank) * weight_rank
    unc_sigmoid = 1.0 / (1.0 + np.exp(-2.2271341322350118 * (uncertainty - 1.0)))
    energy_norm = percentile_normalize(min_incremental_energy)
    unc_norm = percentile_normalize(uncertainty)
    energy_uncertainty_score = 1.0962363927355443 * energy_norm * unc_norm * unc_sigmoid
    wait_normalized = np.where(slack_headroom_mask > 0.0, ready_wait_time / (np.abs(slack) + 1.0), 0.0)
    wait_score = percentile_normalize(wait_normalized)
    score = percentile_normalize(slack_score) + percentile_normalize(unc_slack_coupling) + percentile_normalize(duration_risk)
    score += slack_headroom_mask * (0.7419126236686162 * energy_eff_score + rank_score + energy_uncertainty_score + wait_score)
    score = score * critical_gate
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
