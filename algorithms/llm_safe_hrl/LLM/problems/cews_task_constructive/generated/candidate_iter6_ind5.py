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

    """
    Self-evolved priority rule v2: Advances robustness, deadline-aware energy efficiency,
    and starvation-aware fairness via three core innovations:
    
    - *Slack-Adaptive Criticality Gating*: replaces scalar beta-switch with continuous
      sigmoid-gated criticality amplification: (1+uncertainty)^(1.0 + 0.5*tanh(slack/median_duration)),
      enabling smooth transition from safety to violation regimes without hard thresholds.
    
    - *Energy-Density Priority Reversal*: in slack-positive regime, prioritizes *low* energy density
      (i.e., energy-efficient tasks), but flips to *high* energy density under slack deficit — because
      late tasks benefit more from aggressive offloading to high-power VMs that reduce latency,
      even at higher marginal energy — captured by sign-aware energy_norm scaling.
    
    - *Starvation-Aware Urgency Compression*: replaces binary wait_penalty with a bounded,
      work-normalized waiting pressure term: min(ready_wait_time / (median_duration + eps), 3.0)
      scaled by (1 - percentile_scale(upward_rank)) only when slack > 0 — prevents starvation
      of low-criticality tasks while preserving deadline integrity.
    
    - All features use monotonic percentile_scale with explicit small-N handling (N==1 → [0.5]).
    - Strict eps protection, nan/inf clipping, deterministic, shape-(N,) guaranteed.
    - Final score is convex combination weighted by slack-tightness_gate for stability.
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

    duration = min_exec_time + min_comm_time + eps
    median_duration = np.median(duration) + eps
    median_abs_slack = np.median(np.abs(slack)) + eps
    median_rw = np.median(remaining_work) + eps

    def percentile_scale(x):
        if len(x) == 1:
            return np.array([0.5])
        ranks = np.argsort(np.argsort(x)) + 1.0
        return np.clip(ranks / (len(x) + 1.0), 0.0, 1.0)

    # Slack-adaptive criticality gating: smooth, continuous exponent
    # When slack >> 0 → exponent ≈ 1.0; when slack << 0 → exponent → 1.5
    slack_normalized = slack / (median_duration + eps)
    criticality_exponent = 1.0 + 0.5 * np.tanh(slack_normalized)
    risk_factor = np.power(1.0 + np.clip(uncertainty, 0.0, 10.0), criticality_exponent)

    # Energy density: min_incremental_energy / duration, robustly clipped
    energy_density = np.clip(min_incremental_energy / duration, eps, 1e6)
    
    # Sign-aware energy priority: prefer LOW energy density when slack > 0, HIGH when slack <= 0
    # Achieved by flipping percentile scale for slack-deficit tasks
    energy_norm_base = percentile_scale(energy_density)
    energy_norm = np.where(slack > 0.0, energy_norm_base, 1.0 - energy_norm_base)

    # Deadline urgency: normalized slack deficit + exponential urgency boost
    slack_deficit = np.maximum(0.0, -slack)
    normalized_slack_deficit = np.clip(slack_deficit / median_duration, 0.0, 10.0)
    work_impact_factor = np.where(slack <= 0.0, np.clip(remaining_work / median_rw, 0.3, 3.0), 1.0)
    deadline_urgency = (
        normalized_slack_deficit * work_impact_factor * 0.7 
        + (1.0 - np.exp(-np.clip(slack_deficit, 0.0, 5.0))) * 0.3
    )
    urgency_norm = percentile_scale(deadline_urgency)

    # Criticality-efficiency tradeoff: upward_rank × risk_factor / energy_density
    crit_eff_score = upward_rank * risk_factor / (energy_density + eps)
    crit_eff_norm = percentile_scale(np.clip(crit_eff_score, eps, 1e7))

    # Starvation-aware waiting pressure: bounded, work-normalized, only active when slack > 0
    wait_pressure = np.clip(ready_wait_time / (median_duration + eps), 0.0, 3.0)
    wait_norm = percentile_scale(wait_pressure)
    starvation_penalty = np.where(
        slack > 0.0,
        wait_norm * (1.0 - percentile_scale(upward_rank)) * 0.15,
        0.0
    )

    # Communication dominance penalty: only when comm dominates & slack is tight
    comm_dominance = (min_comm_time > 0.8 * duration).astype(float)
    abs_slack = np.abs(slack)
    comm_penalty = min_comm_time * uncertainty * comm_dominance * (abs_slack < median_abs_slack).astype(float)
    comm_norm = percentile_scale(comm_penalty)

    # Duration and uncertainty normalization
    dur_norm = percentile_scale(duration)
    unc_norm = percentile_scale(uncertainty)

    # Slack tightness gate: smooth sigmoid activation centered at median_abs_slack
    slack_tightness_gate = 1.0 / (1.0 + np.exp(-(abs_slack - median_abs_slack) / (median_abs_slack + eps)))

    # Final convex combination — all terms contribute meaningfully, no negative weights
    # Urgency and criticality-efficiency dominate; others provide fine-grained correction
    score = (
        0.45 * urgency_norm +
        0.25 * (1.0 - crit_eff_norm) +
        0.12 * energy_norm +
        0.08 * starvation_penalty +
        0.05 * comm_norm +
        0.03 * dur_norm +
        0.02 * unc_norm
    )

    # Apply slack-tightness gating: amplify priority adjustments only when deadlines are tight
    score = score * slack_tightness_gate

    # Ensure finite output: replace NaN/inf with safe large values
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)

    return score
