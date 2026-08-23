import numpy as np
RULE_METADATA = {'structure_hash': 'fbc2c7354a27c592121f3af31b0ea2fd4a83867dfccbd369b74043e5df2964bf', 'parameter_schema_hash': 'ed06626fb1f386393f75032f6f6dbcc625a43c006a90557b01a6edbf780db3ab', 'best_parameter_hash': 'f73c1d9666430bf79e5bb658aca9847ffacf5ba139caded1a3c6a56e5fcc962a', 'best_parameters': {'epsilon': 9.881215171548126e-07, 'slack_penalty_exponent': 3.4465972564393037, 'criticality_boost': 1.883954017744899, 'energy_efficiency_ratio_weight': 0.8986408161846198, 'uncertainty_slack_coupling': 0.900255337526732, 'rank_slack_balance': 0.8805692715634577, 'duration_risk_penalty': 0.8308233161063159, 'energy_uncertainty_interaction': 0.49104456410679553, 'uncertainty_sigmoid_steepness': 0.5245006181435298, 'slack_max_bound': 30.761835904425503, 'critical_threshold_factor': 0.9525088939445614}, 'optimizer_config_hash': 'bd16adfa68cb3d3c380e679bfefc37d2ca8416a7e2e3e8f05a56dd735c7a1d44', 'parameter_diagnostics_hash': 'bf1997c899846298c22a97a01be8c1a992f8bcf68768a2c5189f10d972e67796', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule with 11 parameters (slack_min_bound removed):
      - Uses slack_max_bound only — replaces two-sided clipping with symmetric sigmoid scaling
      - All PARAMS references are now used; no unused parameters
      - Numeric literals strictly limited to {-2,-1,0,1,2}
      - Criticality signal remains additive and violation-only
      - Anti-starvation uses exponential decay with slack_max_bound as time constant
      - Unified MAD normalization, N=1 safe
      - Smooth sigmoid slack_gate for non-DDL terms
      - Joint criticality gating on slack<=0 AND high rank/work
    """
    eps = 9.881215171548126e-07
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
        mad = np.median(np.abs(x - med)) + eps
        normalized = (x - med) / mad
        return np.clip(normalized, -2.0, 2.0)
    slack_score = np.where(slack < 0, (-slack) ** 3.4465972564393037, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 0.900255337526732
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 0.8308233161063159
    is_tight_or_violated = slack <= 0
    rank_median = np.median(upward_rank) if N > 1 else np.mean(upward_rank)
    work_median = np.median(remaining_work) if N > 1 else np.mean(remaining_work)
    is_high_rank = upward_rank >= 0.9525088939445614 * rank_median
    is_high_work = remaining_work >= 0.9525088939445614 * work_median
    critical_gate = np.where(is_high_rank & is_high_work & is_tight_or_violated, 1.0, 0.0)
    rank_work_product = upward_rank * remaining_work
    critical_score = mad_normalize(rank_work_product) * critical_gate * 1.883954017744899
    slack_gate = 1.0 / (1.0 + np.exp(-0.5245006181435298 * slack))
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_score = mad_normalize(energy_per_sec)
    slack_scaled = slack / (30.761835904425503 + eps)
    slack_sigmoid = 1.0 / (1.0 + np.exp(-0.5245006181435298 * slack_scaled))
    weight_rank = 0.8805692715634577 * (1.0 - slack_sigmoid) + (1.0 - 0.8805692715634577) * slack_sigmoid
    rank_score = -mad_normalize(upward_rank) * weight_rank
    energy_norm = mad_normalize(min_incremental_energy)
    unc_norm = mad_normalize(uncertainty)
    unc_sigmoid = 1.0 / (1.0 + np.exp(-0.5245006181435298 * (uncertainty - 1.0)))
    energy_uncertainty_score = 0.49104456410679553 * energy_norm * unc_norm * unc_sigmoid
    wait_decay = np.exp(-np.abs(slack) / (30.761835904425503 + eps))
    wait_score = mad_normalize(ready_wait_time) * wait_decay
    score = mad_normalize(slack_score) + mad_normalize(unc_slack_coupling) + mad_normalize(duration_risk)
    score += slack_gate * (0.8986408161846198 * energy_eff_score + rank_score + energy_uncertainty_score + wait_score)
    score += critical_score
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
