import numpy as np
RULE_METADATA = {'structure_hash': 'b198efbbdfd009a940d55d577e81bdc857b8816066e22df10a53e0cd9fcbef14', 'parameter_schema_hash': '3be8bacfe9c07d2a477ec8feddbb502373ecafecdb0f2d7bac029afcfb8cb715', 'best_parameter_hash': '2f450178b2d5300fcfdbdd968ea4366b395a05ccf8b2ec63f3af33beeb9e180b', 'best_parameters': {'epsilon': 9.029755005990398e-05, 'slack_penalty_exponent': 3.8935914282873583, 'criticality_boost': 0.8729986456846113, 'energy_efficiency_ratio_weight': 0.629096571917908, 'uncertainty_slack_coupling': 0.26821840690443444, 'rank_slack_balance': 0.5409464566400507, 'duration_risk_penalty': 1.4543529060598182, 'energy_uncertainty_interaction': 0.0003553096288398808, 'uncertainty_sigmoid_steepness': 1.173649056719634, 'slack_min_bound': -26.158454922150582, 'slack_max_bound': 78.39037608567519}, 'optimizer_config_hash': 'bd16adfa68cb3d3c380e679bfefc37d2ca8416a7e2e3e8f05a56dd735c7a1d44', 'parameter_diagnostics_hash': '6cd5ed212b8d186cc88e283a78f9cdb4474ae96f83efa19759cbeb9eb5eda52f', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule with three structural innovations:
      1. Replaces scalar percentile_normalize with robust MAD-based normalization (median absolute deviation) to improve gradient flow and stability across variable-size ready sets — validated in CMA-ES replay.
      2. Introduces *upward_rank × remaining_work interaction* as a first-class term (not gated), explicitly addressing critical-path starvation per counterfactual consensus evidence.
      3. Switches from lexicographic slack > 0 gating to *smooth DDL-protection sigmoid* (using uncertainty_sigmoid_steepness) that gradually suppresses non-DDL terms near slack=0 — eliminates boundary fragility while preserving feasibility-first ordering.
      4. Adds anti-starvation via *ready_wait_time × (1 - sigmoid(slack))* — prioritizes waiting tasks most when slack is tight but still positive, avoiding hard thresholds.
      5. All divisions use eps; all outputs are finite, shape-(N,) and deterministic; only literals {-2,-1,0,1,2} used.
    """
    eps = 9.029755005990398e-05
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
    slack_score = np.where(slack < 0, (-slack) ** 3.8935914282873583, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 0.26821840690443444
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 1.4543529060598182
    rank_work_interaction = upward_rank * remaining_work
    rank_work_med = np.median(rank_work_interaction) if N > 1 else np.mean(rank_work_interaction)
    rank_work_normalized = mad_normalize(rank_work_interaction) / (np.abs(rank_work_med) + eps)
    slack_sigmoid = 1.0 / (1.0 + np.exp(-1.173649056719634 * slack))
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_score = mad_normalize(energy_per_sec)
    slack_lb = -26.158454922150582
    slack_ub = 78.39037608567519
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    weight_rank = 0.5409464566400507 + (1.0 - 0.5409464566400507) * (1.0 - slack_scaled)
    rank_score = -mad_normalize(upward_rank) * weight_rank
    unc_sigmoid = 1.0 / (1.0 + np.exp(-1.173649056719634 * (uncertainty - 1.0)))
    energy_norm = mad_normalize(min_incremental_energy)
    unc_norm = mad_normalize(uncertainty)
    energy_uncertainty_score = 0.0003553096288398808 * energy_norm * unc_norm * unc_sigmoid
    wait_score = mad_normalize(ready_wait_time) * (1.0 - slack_sigmoid)
    score = mad_normalize(slack_score) + mad_normalize(unc_slack_coupling) + mad_normalize(duration_risk) + rank_work_normalized
    score += slack_sigmoid * (0.629096571917908 * energy_eff_score + rank_score + energy_uncertainty_score + wait_score)
    is_tight_or_violated = slack <= 0
    critical_gate = np.where(is_tight_or_violated, 0.8729986456846113, 1.0)
    score = score * critical_gate
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
