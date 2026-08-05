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
    Hybrid priority rule: hard deadline enforcement + harmonic criticality-energy efficiency +
    adaptive starvation guard + slack-proportional work penalty + latency-criticality synergy.
    
    Key innovations:
    - Combines Parent 2's strict percentile-30 urgency gating with Parent 1's robust harmonic efficiency
      to avoid division instability while preserving criticality-energy tradeoff fidelity.
    - Replaces linear deadline penalty with smoothed but threshold-gated urgency ramp:
      uses clipped tanh(-slack / (duration + eps)) for stable near-zero behavior and bounded range [-1,1].
    - Harmonic criticality-efficiency ratio: 2/(1/ce_ratio + 1/synergy_ratio), avoiding outlier distortion
      and ensuring well-defined scores even when either term dominates.
    - Starvation guard now adaptive *and* uncertainty-aware: wait boost activates only when both
      (ready_wait_time > p75) AND (uncertainty < 0.8), preventing over-prioritization of highly uncertain tasks.
    - Work penalty uses Parent 2's hard-gated slack depletion but scaled by normalized remaining_work
      to penalize heavy subtrees *only* when slack is critically low relative to median slack.
    - All terms independently robust-scaled via median/IQR with MAD fallback; final score bounded & nan-cleaned.
    """
    eps = 1e-08
    # Convert and copy inputs safely
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

    def robust_scale(x):
        if len(x) == 0:
            return x
        med = np.median(x)
        x_centered = x - med
        q1, q3 = np.quantile(x_centered, [0.25, 0.75], method='midpoint')
        iqr = q3 - q1
        if iqr < eps:
            mad = np.median(np.abs(x_centered))
            scale = mad if mad > eps else np.mean(np.abs(x_centered)) + eps
        else:
            scale = iqr + eps
        scaled = x_centered / scale
        return np.clip(scaled, -1000000.0, 1000000.0)

    # Core derived features
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    ce_ratio = upward_rank / (min_incremental_energy + eps)  # criticality per energy
    synergy_ratio = upward_rank * duration / (min_incremental_energy + eps)  # importance × latency per energy
    # Harmonic combination avoids dominance by extreme values in either term
    harmonic_ce = 2.0 / (1.0 / (ce_ratio + eps) + 1.0 / (synergy_ratio + eps))
    harmonic_ce = np.clip(harmonic_ce, eps, 1e6)

    # Urgency: smoothed but threshold-aware — tanh-based for stability & bounded output [-1,1]
    slack_30 = np.quantile(slack, 0.3, method='midpoint') if N > 1 else np.min(slack)
    normalized_urgency_input = -slack / (duration + eps)
    urgency = np.tanh(normalized_urgency_input)  # bounded, smooth, monotonic
    # Hard gating: only amplify urgency where slack < slack_30 (critical region)
    is_critical = slack < (slack_30 - eps)
    urgency_score = np.where(is_critical, urgency * (1.0 + 0.5 * np.clip(uncertainty, 0.0, 2.0)), urgency)

    # Starvation guard: activate only for long-waiting *and* low-uncertainty tasks (avoids noisy scheduling)
    wait_thresh = np.quantile(ready_wait_time, 0.75) if N > 1 else np.max(ready_wait_time)
    starvation_cond = (ready_wait_time > wait_thresh + eps) & (uncertainty < 0.8)
    wait_boost = np.where(starvation_cond, 
                         (ready_wait_time - wait_thresh) / (np.maximum(np.std(ready_wait_time), eps) + eps),
                         0.0)
    wait_score = -wait_boost  # higher wait → lower score (higher priority)

    # Work penalty: only when slack is critically low relative to slack_30, scaled by normalized work
    slack_gap = np.clip(slack_30 - slack, 0.0, None)
    work_norm = robust_scale(remaining_work)
    work_penalty = work_norm * (slack_gap / (np.abs(slack_30) + eps))

    # Energy and latency base terms
    energy_base = min_incremental_energy * (1.0 + 0.7 * np.clip(uncertainty, 0.0, 1.0))
    energy_score = robust_scale(energy_base)
    latency_score = robust_scale(duration)

    # Normalize semantic components
    urgency_norm = robust_scale(urgency_score)
    ce_norm = robust_scale(harmonic_ce)
    ce_score = -ce_norm  # higher harmonic CE → lower score (higher priority)
    synergy_score = robust_scale(synergy_ratio)
    synergy_score = -synergy_score

    # Final weighted score: smaller = higher priority
    score = (
        2.0 * (-urgency_norm) +           # Strong deadline urgency (inverted: urgency_norm high → score low)
        0.4 * ce_score +                 # Criticality-energy efficiency
        0.35 * synergy_score +           # Latency-criticality synergy
        0.25 * energy_score +            # Risk-adjusted energy cost
        0.2 * latency_score +            # Duration penalty
        0.15 * wait_score +              # Starvation mitigation
        0.18 * work_penalty              # Heavy-subtree penalty under urgency
    )

    # Ensure finite, deterministic, shape-compliant output
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    score = np.clip(score, -1e12, 1e12)
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
