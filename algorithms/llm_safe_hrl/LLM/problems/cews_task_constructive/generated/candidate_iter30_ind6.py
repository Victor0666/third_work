import numpy as np

# 函数名中的 2 由 SeEvo 在生成 Prompt 时替换为目标版本号。
# 八个输入均为长度 N 的一维数组；相同下标始终指向同一个 ready task。
# 返回值也必须是长度 N 的一维数组，并遵守“分数越小，优先级越高”。
def get_task_priority_v2(
    min_exec_time,
    min_comm_time,
    min_incremental_energy,
    slack,
    upward_rank,
    remaining_work,
    ready_wait_time,
    uncertainty
):

    '''
    v2 priority rule: Deadline-hardened + risk-conditional energy optimization + 
                      starvation-resilient criticality scaling + 
                      uncertainty-aware fairness gating + 
                      robust zero-slack boundary + 
                      dynamic weight calibration via normalized gradient sensitivity.

    Key improvements over v1:
    - Replaces fixed coefficient weights with *sensitivity-calibrated weights*: 
      computed via finite-difference gradient magnitude of each term w.r.t. slack, 
      ensuring higher weight for terms most responsive to deadline pressure.
    - Introduces *criticality decay* under tight slack: scaled_critical_timing *= (1 - sigmoid(rel_slack + 1)) 
      to further suppress non-urgent long-duration tasks when slack is small but positive.
    - Strengthens fairness by replacing relative_age with *normalized wait-to-critical-ratio*, 
      bounded and clipped to [0,1] to prevent outlier distortion.
    - Adds *uncertainty-fairness coupling*: fairness_boost only activates when uncertainty > median 
      AND slack > eps → avoids boosting starved low-risk tasks.
    - Uses *slack-aligned energy gating*: efficiency_boost now scales with (rel_slack - 0.5) when rel_slack > 0.5, 
      enabling smoother energy preference ramp-up.
    - All normalizations use percentile-clipped minmax with explicit NaN/inf guard + finite-domain fallback.
    - Explicit zero-slack enforcement: lateness_penalty triggers at slack <= 0 (not just eps), 
      preserving hard DDL semantics while keeping epsilon for numeric stability in divisions.
    '''
    eps = 1e-08
    min_exec_time = np.asarray(min_exec_time, dtype=np.float64).copy()
    min_comm_time = np.asarray(min_comm_time, dtype=np.float64).copy()
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=np.float64).copy()
    slack = np.asarray(slack, dtype=np.float64).copy()
    upward_rank = np.asarray(upward_rank, dtype=np.float64).copy()
    remaining_work = np.asarray(remaining_work, dtype=np.float64).copy()
    ready_wait_time = np.asarray(ready_wait_time, dtype=np.float64).copy()
    uncertainty = np.asarray(uncertainty, dtype=np.float64).copy()
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=np.float64)

    def robust_minmax_norm(x):
        if x.size == 0:
            return np.zeros_like(x)
        if N == 1:
            return np.zeros_like(x)
        p01 = np.percentile(x, 1.0, method='lower')
        p99 = np.percentile(x, 99.0, method='higher')
        x_clipped = np.clip(x, p01, p99)
        x_min = np.min(x_clipped)
        x_max = np.max(x_clipped)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        return (x_clipped - x_min) / (x_max - x_min + eps)

    # Hard deadline enforcement: trigger penalty at slack <= 0 (not eps) for semantic correctness
    is_late_or_due = slack <= 0.0
    abs_slack = np.abs(slack)
    lateness_penalty = np.where(is_late_or_due, -1000000000000.0 * (1.0 + 0.1 * abs_slack), 0.0)

    duration = min_exec_time + min_comm_time + eps
    critical_timing = duration * upward_rank
    norm_critical_timing = robust_minmax_norm(critical_timing)

    rel_slack = np.divide(slack, duration, out=np.zeros_like(slack), where=duration != 0)
    rel_slack = np.nan_to_num(rel_slack, nan=0.0, posinf=0.0, neginf=0.0)

    # Criticality decay under tight slack: suppress premature scheduling more aggressively
    # sigmoid(x) = 1/(1+exp(-x)); shift so rel_slack=0 → sigmoid(1)=0.73 → decay ~27%
    critical_decay = 1.0 - 1.0 / (1.0 + np.exp(-(rel_slack + 1.0)))
    slack_scale = np.clip(1.0 - rel_slack, 0.0, 1.0)
    scaled_critical_timing = norm_critical_timing * slack_scale * critical_decay

    energy_density = np.divide(min_incremental_energy, duration, out=np.zeros_like(min_incremental_energy), where=duration != 0)
    energy_density = np.nan_to_num(energy_density, nan=0.0, posinf=0.0, neginf=0.0)
    norm_energy_density = robust_minmax_norm(energy_density)

    # Slack-aligned energy boost: linear ramp from 0 at rel_slack=0.5 to 0.3 at rel_slack=1.0
    efficiency_ramp = np.clip((rel_slack - 0.5) * 0.6, 0.0, 0.3)
    efficiency_boost_mask = (slack > eps) & (rel_slack > 0.5)
    efficiency_boost = -0.25 * norm_energy_density * efficiency_boost_mask * efficiency_ramp

    # Wait-to-critical-ratio: normalized starvation pressure bounded in [0,1]
    cp_estimate = duration * upward_rank + eps
    wait_ratio = np.divide(ready_wait_time, cp_estimate, out=np.zeros_like(ready_wait_time), where=cp_estimate != 0)
    wait_ratio = np.clip(np.nan_to_num(wait_ratio, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    norm_wait_ratio = robust_minmax_norm(wait_ratio)

    rw_median = np.median(remaining_work) if N > 1 else remaining_work[0]
    ur_median = np.median(upward_rank) if N > 1 else upward_rank[0]
    unc_median = np.median(uncertainty) if N > 1 else uncertainty[0]

    # Uncertainty-aware fairness: only boost if both slack > 0 AND uncertainty > median
    fairness_mask = (slack > eps) & (remaining_work > rw_median) & (uncertainty > unc_median)
    fairness_boost = norm_wait_ratio * fairness_mask

    # Uncertainty coupling: triple condition remains, but now uses norm_uncertainty directly (not redundant product)
    unc_mask = (slack > eps) & (upward_rank > ur_median) & (uncertainty > unc_median)
    norm_uncertainty = robust_minmax_norm(uncertainty)
    unc_coupling = norm_uncertainty * unc_mask

    norm_remaining_work = robust_minmax_norm(remaining_work)

    # Sensitivity-calibrated weights via finite-difference gradient magnitude w.r.t. slack
    # Approximate d(term)/d(slack) using central difference with small perturbation
    slack_perturb = np.where(np.abs(slack) < 1e-3, 1e-3, np.abs(slack) * 0.01)
    slack_plus = slack + slack_perturb
    slack_minus = slack - slack_perturb

    # Compute gradient magnitude for each component (only where defined)
    rel_slack_plus = np.divide(slack_plus, duration, out=np.zeros_like(slack), where=duration != 0)
    rel_slack_minus = np.divide(slack_minus, duration, out=np.zeros_like(slack), where=duration != 0)
    rel_slack_plus = np.nan_to_num(rel_slack_plus, nan=0.0, posinf=0.0, neginf=0.0)
    rel_slack_minus = np.nan_to_num(rel_slack_minus, nan=0.0, posinf=0.0, neginf=0.0)

    # Gradient of scaled_critical_timing w.r.t slack (dominant term)
    slack_scale_plus = np.clip(1.0 - rel_slack_plus, 0.0, 1.0)
    slack_scale_minus = np.clip(1.0 - rel_slack_minus, 0.0, 1.0)
    critical_decay_plus = 1.0 - 1.0 / (1.0 + np.exp(-(rel_slack_plus + 1.0)))
    critical_decay_minus = 1.0 - 1.0 / (1.0 + np.exp(-(rel_slack_minus + 1.0)))
    grad_crit = np.abs((norm_critical_timing * slack_scale_plus * critical_decay_plus - 
                        norm_critical_timing * slack_scale_minus * critical_decay_minus) / (2 * slack_perturb + eps))

    # Gradient of energy_penalty surrogate (here: norm_energy_density * slack_proximity approximated)
    slack_prox_plus = np.clip(0.3 - rel_slack_plus, 0.0, 0.3)
    slack_prox_minus = np.clip(0.3 - rel_slack_minus, 0.0, 0.3)
    grad_energy = np.abs((norm_energy_density * slack_prox_plus - norm_energy_density * slack_prox_minus) / (2 * slack_perturb + eps))

    # Gradient of fairness_boost
    fairness_mask_plus = (slack_plus > eps) & (remaining_work > rw_median) & (uncertainty > unc_median)
    fairness_mask_minus = (slack_minus > eps) & (remaining_work > rw_median) & (uncertainty > unc_median)
    grad_fair = np.abs((norm_wait_ratio * fairness_mask_plus - norm_wait_ratio * fairness_mask_minus) / (2 * slack_perturb + eps))

    # Aggregate sensitivities and normalize to convex weights
    sens = np.stack([grad_crit, grad_energy, grad_fair, np.ones_like(grad_crit)], axis=0)
    sens_sum = np.sum(sens, axis=0) + eps
    w_crit = (grad_crit / sens_sum)
    w_energy = (grad_energy / sens_sum)
    w_fair = (grad_fair / sens_sum)
    w_rem = (np.ones_like(grad_crit) / sens_sum)

    # Normalize weights to sum to 1.0 and clamp extremes
    w_total = w_crit + w_energy + w_fair + w_rem
    w_crit = np.clip(w_crit / (w_total + eps), 0.2, 0.6)
    w_energy = np.clip(w_energy / (w_total + eps), 0.05, 0.25)
    w_fair = np.clip(w_fair / (w_total + eps), 0.05, 0.25)
    w_rem = 1.0 - w_crit - w_energy - w_fair
    w_rem = np.clip(w_rem, 0.05, 0.3)

    # Final score with sensitivity-weighted components
    score = (w_crit * scaled_critical_timing + 
             w_energy * (robust_minmax_norm(uncertainty) * unc_mask) + 
             w_fair * fairness_boost + 
             w_rem * norm_remaining_work + 
             efficiency_boost)

    score = lateness_penalty + score

    # Final sanitization
    score = np.nan_to_num(score, nan=1000000000000.0, posinf=1000000000000.0, neginf=-1000000000000.0)
    score = np.clip(score, -1000000000000.0, 1000000000000.0)

    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
