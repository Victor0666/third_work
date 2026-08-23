import numpy as np
RULE_METADATA = {'structure_hash': '9aa410d76a9347e37e3969c8b6571e0875f6941275f483ee3ab8d6b3a6adde4e', 'parameter_schema_hash': '14df6999c45fc3c5cbf3f1c9dba134b5e7443dc0e5a10b1fa17904d93c0c9e8e', 'best_parameter_hash': '1447a90d8045ba222ddd0f141f37e12fac44fc9c95b0af8cb606a29c325f4662', 'best_parameters': {'epsilon': 3.861799633977871e-07, 'slack_penalty_exponent': 1.7185869263167792, 'criticality_boost': 3.129000910177622, 'energy_efficiency_ratio_weight': 1.238299799426568, 'uncertainty_slack_coupling': 0.7390888532267224, 'rank_slack_balance': 0.8840915206812443, 'duration_risk_penalty': 1.5399074455174662, 'energy_uncertainty_interaction': 0.8942647746113798, 'uncertainty_sigmoid_steepness': 2.4443583088260707, 'slack_min_bound': -18.458229896391586, 'slack_max_bound': 58.10494488857636, 'percentile_clip_low': 16.390623945797262}, 'optimizer_config_hash': 'bd16adfa68cb3d3c380e679bfefc37d2ca8416a7e2e3e8f05a56dd735c7a1d44', 'parameter_diagnostics_hash': '9de66995d0ac0ec90eb9365b1563077f5c23b381fe33b12692f8c562222debe9', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's strict lexicographic DDL enforcement and clipped percentile normalization
    with Parent 1's successor-release urgency and starvation mitigation:
      - Strict lexicographic gate: non-DDL terms active only when slack > 0.0 (DDL safety first)
      - Criticality boost jointly conditioned on slack <= 0, high upward_rank, AND high remaining_work
      - Successor-release term: upward_rank * (1 + remaining_work/(|slack|+eps)) activated under DDL pressure
      - Wait-time anti-starvation uses `ready_wait_time / (|slack| + 1)` only when slack > 0, then percentile-normalized
      - All normalization uses clipped percentile (q_low, symmetric q_high = 2*med − q_low)
      - No numeric literals beyond {-2,-1,0,1,2}; all tunables declared and used exactly once.
    """
    eps = 3.861799633977871e-07
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
        q_low = np.percentile(x, 16.390623945797262)
        med = np.median(x)
        q_high = 2.0 * med - q_low
        denom = q_high - q_low + eps
        normalized = (x - q_low) / denom
        return np.clip(normalized, 0.0, 1.0)
    slack_score = np.where(slack < 0, (-slack) ** 1.7185869263167792, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 0.7390888532267224
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 1.5399074455174662
    is_tight_or_violated = slack <= 0
    rank_median = np.median(upward_rank) if N > 1 else np.mean(upward_rank)
    work_median = np.median(remaining_work) if N > 1 else np.mean(remaining_work)
    is_high_rank = upward_rank >= rank_median
    is_high_work = remaining_work >= work_median
    critical_gate = np.where(is_high_rank & is_high_work & is_tight_or_violated, 3.129000910177622, 1.0)
    abs_slack_safe = np.abs(slack) + eps
    successor_release_gain = 1.0 + remaining_work / abs_slack_safe
    successor_urgency = upward_rank * successor_release_gain
    slack_headroom_mask = np.where(slack > 0.0, 1.0, 0.0)
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_score = percentile_normalize(energy_per_sec)
    slack_lb = -18.458229896391586
    slack_ub = 58.10494488857636
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    weight_rank = 0.8840915206812443 + (1.0 - 0.8840915206812443) * (1.0 - slack_scaled)
    rank_score = -percentile_normalize(upward_rank) * weight_rank
    unc_sigmoid = 1.0 / (1.0 + np.exp(-2.4443583088260707 * (uncertainty - 1.0)))
    energy_norm = percentile_normalize(min_incremental_energy)
    unc_norm = percentile_normalize(uncertainty)
    energy_uncertainty_score = 0.8942647746113798 * energy_norm * unc_norm * unc_sigmoid
    wait_normalized = np.where(slack_headroom_mask > 0.0, ready_wait_time / (np.abs(slack) + 1.0), 0.0)
    wait_score = percentile_normalize(wait_normalized)
    score = percentile_normalize(slack_score) + percentile_normalize(unc_slack_coupling) + percentile_normalize(duration_risk) + percentile_normalize(successor_urgency)
    score += slack_headroom_mask * (1.238299799426568 * energy_eff_score + rank_score + energy_uncertainty_score + wait_score)
    score = score * critical_gate
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
