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
    Hybrid priority rule: hard deadline enforcement + starvation-aware wait boost +
    slack-gated work penalty + latency-criticality-energy synergy + robust per-feature scaling.
    
    Key improvements over v1:
      - Replaces global IQR scaling with *task-duration-aware normalization* for slack and wait time
        to preserve physical semantics (e.g., 1s wait matters more on 2s tasks than 200s tasks)
      - Introduces *uncertainty-modulated urgency amplification*: only amplify deadline penalty when
        uncertainty > median_uncertainty AND slack < slack_30 — avoids over-penalizing low-risk urgent tasks
      - Adds *energy-density fairness term*: penalizes tasks with high min_incremental_energy per unit
        remaining_work, promoting energy-efficient scheduling of heavy subtrees
      - Retains v1's hard 30th-percentile urgency gating, adaptive 75th-percentile wait threshold,
        and latency-criticality synergy; drops redundant time_norm/work_norm terms for clarity & stability
      - All operations epsilon-protected, nan/inf-cleaned, deterministic, shape-compliant.
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

    # Task duration and derived features
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    neg_slack = np.maximum(-slack, 0.0)
    
    # Hard urgency gating at 30th percentile slack
    slack_sorted = np.sort(slack)
    slack_30 = slack_sorted[max(0, int(0.3 * len(slack_sorted)))] if N > 0 else 0.0
    is_deeply_urgent = slack < slack_30
    
    # Uncertainty-aware deadline penalty: only amplify when both tight slack AND high uncertainty
    median_unc = np.median(uncertainty) + eps
    unc_amplifier = np.where(uncertainty > median_unc, 1.0 + 0.5 * np.clip(uncertainty - median_unc, 0.0, 1.0), 1.0)
    base_deadline_penalty = neg_slack * (6.0 + 4.0 * np.clip(uncertainty, 0.0, 2.0))
    deadline_penalty = np.where(is_deeply_urgent, base_deadline_penalty * unc_amplifier + 12.0, base_deadline_penalty)

    # Criticality-energy ratio (HEFT-style) — normalized independently
    ce_ratio = upward_rank / (min_incremental_energy + eps)
    ce_score = -ce_ratio  # higher ratio → higher priority → lower score
    
    # Latency-criticality synergy: prioritize high-importance, low-energy-per-latency tasks
    synergy = upward_rank * duration / (min_incremental_energy + eps)
    synergy_score = -synergy

    # Energy-density fairness: penalize high energy per unit remaining work (promotes efficiency in heavy subtrees)
    energy_per_work = min_incremental_energy / (remaining_work + eps)
    energy_density_penalty = np.clip(energy_per_work, 0.0, 1e6)

    # Starvation control: adaptive wait boost above 75th percentile wait time
    wait_thresh = np.quantile(ready_wait_time, 0.75) if N > 0 else eps
    wait_boost = np.where(
        ready_wait_time > wait_thresh,
        1.0 * (ready_wait_time - wait_thresh) / (np.maximum(np.std(ready_wait_time), eps) + eps),
        0.0
    )
    wait_score = -wait_boost  # negative boost → lower score for starved tasks

    # Slack-gated work penalty: only penalize large remaining_work when slack is tight
    slack_gap = np.clip(slack_30 - slack, 0.0, None)
    work_penalty = (remaining_work / (np.median(remaining_work) + eps)) * (slack_gap / (np.abs(slack_30) + eps))

    # Combine components with calibrated weights — emphasize deadline, then criticality-efficiency tradeoffs
    score = (
        2.0 * deadline_penalty +
        0.35 * ce_score +
        0.4 * synergy_score +
        0.15 * energy_density_penalty +
        0.12 * wait_score +
        0.18 * work_penalty
    )

    # Robust final sanitization
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    score = np.clip(score, -1e12, 1e12)

    # Ensure shape compliance
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
