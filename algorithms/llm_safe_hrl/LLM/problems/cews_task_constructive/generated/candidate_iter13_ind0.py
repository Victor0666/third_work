import numpy as np

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
    """
    Mutated priority rule v2: emphasizes *slack-margin-aware criticality gating*,
    *uncertainty-calibrated energy dominance*, and *wait-aware fairness with decay*.
    
    Key mutations vs v1:
    - Replaces relative slack ratio with *slack margin ratio*: (deadline - est_finish) / (est_finish + eps),
      computed via robust slack-to-duration alignment using upward_rank-weighted path estimate.
    - Introduces *criticality-gated energy penalty*: only applies strong energy normalization when
      upward_rank > 0.75 * median(upward_rank) AND slack > 0 — avoids penalizing energy under urgency.
    - Replaces adaptive waiting penalty with *exponential decay fairness term*: 
      exp(-ready_wait_time / (median(min_exec_time + min_comm_time) + eps)) inverted to boost long-wait,
      bounded and scaled by remaining_work to prevent trivial-task starvation.
    - Uses *quantile-symmetric energy scaling*: min_incremental_energy normalized via IQR *and*
      re-centered to [0,1] using robust_minmax with 5%/95% quantiles for stable monotonic ranking.
    - Adds *uncertainty-constrained criticality*: upward_rank is dampened by (1 + uncertainty)^beta where
      beta = clip(0.5 * (1 - slack / (|slack| + task_min_duration + eps)), 0.0, 1.0) — reduces rank noise
      under high uncertainty unless slack is severely negative.
    - All terms fused via convex combination (weights sum to 1.0) instead of linear sum, ensuring
      scale-invariant contribution balance; no raw coefficient tuning outside [0,1].
    """
    eps = 1e-08
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=float)

    # Robust quantile-based normalization with 5%/95% bounds and centering
    def robust_minmax(x):
        q05 = np.quantile(x, 0.05)
        q95 = np.quantile(x, 0.95)
        rng = np.maximum(q95 - q05, eps)
        centered = x - (q05 + q95) / 2.0
        return np.clip(centered / rng, -5.0, 5.0)

    # Task intrinsic time cost
    task_min_duration = np.maximum(min_exec_time + min_comm_time, eps)
    
    # Slack margin ratio: (slack) / (task_min_duration + upward_rank contribution)
    # Upward rank approximates remaining critical path; use it to scale deadline sensitivity
    path_estimate = task_min_duration + 0.3 * np.maximum(upward_rank, eps)
    slack_margin_ratio = slack / (path_estimate + eps)
    
    # Hard urgency override: negative slack → dominant priority boost (inverse score)
    # Use smooth sigmoid-like penalty: 1/(1+exp(-x)) maps negative slack to >0.5, positive to <0.5
    urgency_score = 1.0 / (1.0 + np.exp(-np.clip(-slack_margin_ratio, -10.0, 10.0)))
    
    # Uncertainty-constrained criticality: dampen upward_rank under high uncertainty unless urgent
    slack_abs = np.abs(slack) + eps
    beta = np.clip(0.5 * (1.0 - slack / (slack_abs + task_min_duration)), 0.0, 1.0)
    dampened_ur = upward_rank * np.power(1.0 + uncertainty, -beta)
    
    # Criticality-energy coupling only when not urgent (slack > 0) and high criticality
    median_ur = np.median(upward_rank) + eps
    high_crit_mask = (upward_rank > 0.75 * median_ur).astype(float)
    energy_active_mask = (slack > 0.0).astype(float)
    energy_coupling_mask = high_crit_mask * energy_active_mask
    
    # Robust-minmax normalized energy (0–1 range after clipping)
    energy_norm_raw = robust_minmax(min_incremental_energy)
    energy_norm = np.clip((energy_norm_raw + 5.0) / 10.0, eps, 1.0 - eps)  # [0,1]
    
    # Criticality-efficiency ratio: only active when both conditions hold
    crit_eff_ratio = np.where(
        energy_coupling_mask > 0.0,
        dampened_ur / (energy_norm + eps),
        dampened_ur  # fallback: pure criticality when energy coupling inactive
    )
    crit_eff_norm = robust_minmax(crit_eff_ratio)
    
    # Exponential decay fairness: longer wait → higher priority, but capped and work-scaled
    median_path = np.median(task_min_duration) + eps
    wait_decay = np.exp(-np.clip(ready_wait_time / median_path, 0.0, 10.0))
    # Invert and scale: 1-exp(-t) grows from 0→1; multiply by work to prioritize non-trivial waits
    wait_fairness = (1.0 - wait_decay) * np.clip(remaining_work / (np.median(remaining_work) + eps), 0.1, 5.0)
    wait_score = np.clip(wait_fairness, 0.0, 0.3)  # bounded contribution
    
    # Uncertainty as soft priority modifier: higher uncertainty → slightly earlier scheduling
    # but only when slack is non-negative (no interference with hard deadlines)
    unc_score = np.where(slack > 0.0, np.clip(uncertainty, 0.0, 1.0), 0.0)
    
    # Work-normalized urgency: remaining_work scales slack sensitivity for large jobs
    rw_norm = np.clip(remaining_work / (np.median(remaining_work) + eps), 0.1, 10.0)
    slack_weighted = slack_margin_ratio * np.sqrt(rw_norm)  # larger jobs get stronger slack signal
    
    # Convex combination (weights sum to 1.0) — ensures stable contribution balance
    w_urgency = 0.42
    w_criteff = 0.28
    w_wait = 0.15
    w_unc = 0.08
    w_slackwork = 0.07
    
    score = (
        w_urgency * (1.0 - urgency_score) +           # smaller = more urgent → higher priority
        w_criteff * (1.0 - np.clip((crit_eff_norm + 5.0) / 10.0, 0.0, 1.0)) +
        w_wait * (1.0 - wait_score) +
        w_unc * (1.0 - unc_score) +
        w_slackwork * (1.0 - np.clip((slack_weighted + 5.0) / 10.0, 0.0, 1.0))
    )
    
    # Final safeguard: ensure finite, shape-(N,), deterministic output
    score = np.nan_to_num(score, nan=1e6, posinf=1e6, neginf=-1e6)
    return score
