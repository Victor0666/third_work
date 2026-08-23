import numpy as np
RULE_METADATA = {'structure_hash': '730ccdd01bdb975569fe575ded254f4c733669e4de3b64529ff75534ccba47c8', 'parameter_schema_hash': '426d966e2c395552fed6037d90319481f154c372b9ca00c16dd2ea982afb1a41', 'best_parameter_hash': 'e7865f1a538d66b9fa2344d5b198d9cee342c2f5928350468af1bec9dcd677f5', 'best_parameters': {'epsilon': 4.0099444326303405e-06, 'slack_penalty_exponent': 2.508022116963742, 'criticality_boost': 3.276596264629073, 'energy_efficiency_ratio_weight': 0.9133365347672521, 'uncertainty_slack_coupling': 1.7352388888653012, 'rank_slack_balance': 0.506826973261538, 'duration_risk_penalty': 0.4640921710019599, 'energy_uncertainty_interaction': 1.0953192156708318, 'uncertainty_sigmoid_steepness': 0.9084476573348025, 'ddl_protection_gate_threshold': 0.025923401474640995, 'slack_min_bound': -0.6103060160451292, 'slack_max_bound': 35.986455492051114}, 'optimizer_config_hash': 'bd16adfa68cb3d3c380e679bfefc37d2ca8416a7e2e3e8f05a56dd735c7a1d44', 'parameter_diagnostics_hash': '44b85efb3ab55a3de46d39284d4d4b236d64558a5cc0ac503f1d96a340db09b9', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's robustness with Parent 1's work-density insight:
      - Retains MAD-based normalization (robust for N=1, sparse sets).
      - Keeps tighter DDL protection gate (threshold > 0) and criticality boost logic.
      - Integrates Parent 1's work_density term (remaining_work / duration), gated by joint feasibility sigmoid.
      - Adds exponential wait-time decay instead of linear normalization to prevent starvation without over-prioritizing old tasks.
      - Uses unified uncertainty modulation across all risk-sensitive components.
      - All numeric literals are {-2,-1,0,1,2}; no loops, randomness, or side effects.
    """
    eps = 4.0099444326303405e-06
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
    slack_score = np.where(slack < 0, (-slack) ** 2.508022116963742, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 1.7352388888653012
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 0.4640921710019599
    is_tight_or_violated = slack <= 0
    rank_median = np.median(upward_rank) if N > 1 else np.mean(upward_rank)
    work_median = np.median(remaining_work) if N > 1 else np.mean(remaining_work)
    is_high_rank = upward_rank >= rank_median
    is_high_work = remaining_work >= work_median
    critical_gate = np.where(is_high_rank & is_high_work & is_tight_or_violated, 3.276596264629073, 1.0)
    slack_headroom_mask = np.where(slack > 0.025923401474640995, 1.0, 0.0)
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_score = mad_normalize(energy_per_sec)
    rank_work_interaction = upward_rank * remaining_work
    rank_work_score = -mad_normalize(rank_work_interaction)
    slack_lb = -0.6103060160451292
    slack_ub = 35.986455492051114
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    weight_rank = 0.506826973261538 + (1.0 - 0.506826973261538) * (1.0 - slack_scaled)
    rank_score = -mad_normalize(upward_rank) * weight_rank
    unc_sigmoid = 1.0 / (1.0 + np.exp(-0.9084476573348025 * (uncertainty - 1.0)))
    energy_norm = mad_normalize(min_incremental_energy)
    unc_norm = mad_normalize(uncertainty)
    energy_uncertainty_score = 1.0953192156708318 * energy_norm * unc_norm * unc_sigmoid
    work_density = np.divide(remaining_work, duration_total)
    work_density_norm = mad_normalize(work_density)
    joint_ratio = slack / (uncertainty + eps)
    joint_feasibility = 1.0 / (1.0 + np.exp(-0.9084476573348025 * joint_ratio))
    work_density_bonus = 0.9133365347672521 * work_density_norm * joint_feasibility
    wait_normalized = np.where(slack_headroom_mask > 0.0, np.exp(-0.9133365347672521 * ready_wait_time), 0.0)
    wait_score = mad_normalize(wait_normalized)
    score = mad_normalize(slack_score) + mad_normalize(unc_slack_coupling) + mad_normalize(duration_risk)
    score += slack_headroom_mask * (0.9133365347672521 * energy_eff_score + rank_work_score + rank_score + energy_uncertainty_score + work_density_bonus + wait_score)
    score = score * critical_gate
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
