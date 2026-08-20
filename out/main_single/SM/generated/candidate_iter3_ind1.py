import numpy as np
RULE_METADATA = {'structure_hash': '5bd0f64000411a2a4c21653be489a109f5749700b0ac0a6096802dd4dd37dc70', 'parameter_schema_hash': 'c12f3e43025ab52b42f6a00115428dd38bde173a497ec8de8eb5fd0a4eae46b0', 'best_parameter_hash': '1cd922c3ff570219b2c7f07c6b55ad707280ec9e5d0802ab3d40620b9ed643c3', 'best_parameters': {'epsilon': 0.0004104035529410392, 'slack_risk_penalty': 7.192739551102237, 'slack_urgency_gain': 1.973353029743548, 'energy_efficiency_weight': 2.0029504613214124, 'criticality_weight': 0.282401856172492, 'duration_uncertainty_ratio': 0.7226281205403204, 'wait_clip_threshold': 65.24025198675791, 'uncertainty_slack_interaction': 1.5141142371804084, 'successor_work_coupling': 0.45955621997857954, 'percentile_low': 13.725017189059873, 'percentile_high': 85.1661174852617, 'ddl_protection_gate': 1.2445673678623328}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': 'f039efbcb2efb58bb09ff816a8e57403f14637878f034be83780d8cebe1c3833', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with bounded percentile normalization, hard DDL-protection gate,
    and clipped linear starvation mitigation — eliminating fragile exponential decay and outlier-sensitive MAD.
    
    Key self-evolution improvements:
      - Bounded percentile-based normalization (p10–p90) replaces MAD: robust to skew, preserves rank order,
        avoids median instability on small N, and eliminates tuning of scale/mad multipliers.
      - Hard DDL-protection gate: multiplies slack_penalty by ddl_protection_gate when slack < -eps,
        strictly prioritizing deadline compliance over all other objectives under violation risk.
      - Clipped linear wait-time: min(ready_wait_time, wait_clip_threshold) — removes exponential decay
        sensitivity, guarantees bounded contribution, and simplifies starvation control.
      - All components are finite, deterministic, and shaped strictly (N,).
    """
    eps = 0.0004104035529410392
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    work = np.asarray(remaining_work, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)

    def percentile_normalize(x):
        x = np.asarray(x)
        p_low = np.percentile(x, 13.725017189059873)
        p_high = np.percentile(x, 85.1661174852617)
        rng = p_high - p_low + eps
        centered = x - (p_low + p_high) / 2.0
        return np.clip(centered / rng, -2.0, 2.0)
    duration = exec_t + comm_t + 0.45955621997857954 * work
    norm_duration = percentile_normalize(duration + eps)
    inv_energy = 1.0 / (energy + eps)
    norm_energy = percentile_normalize(inv_energy)
    energy_score = -2.0029504613214124 * norm_energy
    norm_slack = percentile_normalize(slk)
    base_slack_penalty = np.where(slk < 0, 7.192739551102237 * norm_slack ** 2, -1.973353029743548 * np.abs(norm_slack))
    slack_penalty = np.where(slk < -eps, 1.2445673678623328 * base_slack_penalty, base_slack_penalty)
    rank_active = np.where(slk >= -eps, rank, 0.0)
    norm_rank = percentile_normalize(rank_active + eps)
    rank_score = -0.282401856172492 * norm_rank
    norm_uncert = percentile_normalize(uncert + eps)
    dur_uncert_blend = (1.0 + 0.7226281205403204 * norm_uncert) / (np.abs(norm_duration) + eps + 0.7226281205403204 * norm_uncert + eps)
    dur_score = percentile_normalize(dur_uncert_blend)
    clipped_wait = np.minimum(wait, 65.24025198675791)
    norm_wait = percentile_normalize(clipped_wait + eps)
    wait_score = -norm_wait
    slack_stress = np.where(slk < 0, np.abs(norm_slack), 0.0)
    unc_slack_interaction = 1.5141142371804084 * norm_uncert * slack_stress
    score = slack_penalty + energy_score + rank_score + dur_score + wait_score + unc_slack_interaction
    score = np.nan_to_num(score, nan=np.finfo(float).smallest_subnormal, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
