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
    Hybrid priority rule: urgency-gated dominance with risk-magnitude-aware scaling,
    robust criticality-energy tradeoff, starvation-aware fairness, and slack-proportional work penalty.
    
    Key innovations:
      - Combines Parent 2's sigmoidal urgency gating (for hard DDL compliance) with Parent 1's 
        percentile-based urgency threshold (30th percentile) for sharper early intervention.
      - Uses *dual urgency signals*: hard gate (binary) for deadline violation prevention + soft gate 
        (sigmoid) for smooth prioritization within feasible region.
      - Criticality-energy term uses robust rank-normalized ratio scaled only under urgency > 0.3,
        but adds slack-decay factor exp(-max(0, -slack)/eps) to further suppress non-urgent tasks.
      - Energy penalty now includes uncertainty-weighted slack sensitivity: higher uncertainty amplifies 
        energy penalty *only when slack is tight* (via linear slack-magnitude risk score).
      - Starvation guard activated only for non-urgent (urgency < 0.7) AND non-late (slack >= 0) tasks,
        using wait-age gating (wait > 5% of max_wait) to avoid premature boosting.
      - Introduces *slack-proportional work penalty*: penalizes high remaining_work more severely 
        as slack depletes, using smooth clamp to avoid discontinuities.
      - All components normalized via robust min-max with constant-array safety; final score bounded and finite.
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

    def robust_minmax_norm(x):
        x_min, x_max = np.min(x), np.max(x)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        return (x - x_min) / (x_max - x_min + eps)

    # Compute robust median and IQR for slack
    median_slack = np.median(slack)
    q1_slack, q3_slack = np.percentile(slack, [25, 75])
    iqr_slack = q3_slack - q1_slack + eps

    # Hard urgency gate: top 30% most at-risk tasks (Parent 1 strength)
    slack_sorted = np.sort(slack)
    urgency_thresh_30 = slack_sorted[max(0, int(0.3 * len(slack_sorted)))]
    hard_urgent = (slack <= urgency_thresh_30).astype(float)

    # Soft urgency gate: sigmoid centered at median, scaled by IQR (Parent 2 strength)
    soft_urgency = 1.0 / (1.0 + np.exp(-(slack - median_slack) / (iqr_slack + eps)))

    # Criticality-per-energy: only active for urgent tasks (soft gate > 0.3), with slack decay
    crit_per_energy = upward_rank / (min_incremental_energy + eps)
    norm_crit_per_energy = robust_minmax_norm(crit_per_energy)
    urgency_gate_crit = np.where(soft_urgency > 0.3, 1.0, 0.0)
    # Slack decay: suppress non-urgent via exp(-max(0, -slack)/eps), avoiding overflow
    slack_decay = np.exp(-np.clip(np.maximum(-slack, 0.0), 0.0, 100.0) / (eps + np.abs(median_slack)))
    crit_term = (1.0 - norm_crit_per_energy) * urgency_gate_crit * slack_decay

    # Energy penalty: uncertainty-weighted, amplified under tight slack
    norm_energy = robust_minmax_norm(min_incremental_energy)
    # Risk score: linear from median down to min slack (captures severity, not just sign)
    slack_range = np.maximum(np.abs(np.min(slack) - median_slack), eps)
    risk_score = np.clip((median_slack - slack) / slack_range, 0.0, 1.0)
    energy_penalty = norm_energy * (1.0 + 0.5 * risk_score * uncertainty)

    # Latency term: normalized sum of exec + comm time
    total_latency = min_exec_time + min_comm_time
    norm_latency = robust_minmax_norm(total_latency)

    # Starvation guard: only for non-urgent & non-late tasks, with wait-age gating
    wait_gate = np.where((soft_urgency < 0.7) & (slack >= 0), 1.0, 0.0)
    max_wait = np.max(ready_wait_time) if N > 0 else eps
    wait_norm = np.maximum(max_wait, eps)
    wait_rel = ready_wait_time / wait_norm
    wait_age_gate = (wait_rel > 0.05).astype(float)
    wait_boost = ready_wait_time * wait_gate * wait_age_gate
    norm_wait_boost = robust_minmax_norm(wait_boost) if N > 0 else np.zeros(N)
    starvation_term = 1.0 - norm_wait_boost

    # Slack-proportional work penalty: penalize heavy subtrees more as slack depletes
    norm_work = robust_minmax_norm(remaining_work)
    slack_ratio = np.clip(slack / (np.abs(median_slack) + eps), -5.0, 5.0)
    work_penalty = norm_work * np.maximum(0.0, 1.0 - slack_ratio)

    # Base urgency term: hard gate dominates violations; soft gate refines within feasible set
    base_urgency = 1.0 - soft_urgency
    # Combine all terms with calibrated weights
    score = (
        0.40 * base_urgency +           # Primary urgency enforcement (hard constraint)
        0.22 * energy_penalty +        # Risk-adjusted energy penalty
        0.12 * norm_latency +          # Latency minimization
        0.10 * crit_term +             # Urgency-gated criticality-energy tradeoff
        0.06 * starvation_term +       # Fairness for long-waiting non-urgent tasks
        0.05 * work_penalty +          # Slack-proportional subtree work penalty
        0.05 * (1.0 - hard_urgent)     # Secondary signal: non-urgent tasks get mild baseline boost
    )

    # Ensure finite output and handle edge cases
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    return score
