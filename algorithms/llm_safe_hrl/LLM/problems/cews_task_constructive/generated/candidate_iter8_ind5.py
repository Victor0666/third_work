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
    Hybrid priority rule v2: Combines Parent 2's hard deadline gating and robust per-feature scaling
    with Parent 1's starvation-aware decay, latency-criticality coupling, and risk-congruent uncertainty modulation.
    Key innovations:
      - Hard urgency gating via 30th-percentile slack threshold (Parent 2) + asymmetric penalty amplification for deep lateness
      - Robust independent scaling per physical dimension (Parent 2) to preserve signal fidelity
      - Starvation mitigation with upward-rank-decayed wait boost (Parent 1), activated only when not urgent
      - Latency-criticality synergy term weighted by communication time (Parent 1 enhancement of Parent 2's synergy)
      - Risk-congruent uncertainty boosting: only amplifies uncertainty impact when slack < 0, attenuated smoothly otherwise
      - Energy efficiency gating: penalizes only above-median energy density, avoiding spurious penalties on inherently efficient tasks
      - Strict dominance hierarchy enforced via coefficient weighting: urgency >> synergy > energy > latency > fairness
    """
    eps = 1e-8
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

    # Robust per-feature scaling function preserving physical semantics and sign
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
        # Clamp to avoid overflow while preserving ordering
        return np.clip(scaled, -1e6, 1e6)

    # Core physical quantities
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    neg_slack = np.maximum(-slack, 0.0)
    
    # Hard urgency gating: 30th percentile threshold (Parent 2)
    slack_sorted = np.sort(slack)
    slack_30 = slack_sorted[max(0, int(0.3 * len(slack_sorted)))] if N > 0 else 0.0
    is_deeply_urgent = slack < slack_30
    
    # Asymmetric deadline penalty: steep rise for negative slack, shallow decay for positive
    # Amplified by uncertainty only when lateness risk exists (Parent 1 risk-congruence)
    uncertainty_mod = np.where(slack < 0, 
                              1.0 + 0.8 * np.clip(uncertainty, 0.0, 2.0),
                              np.clip(1.0 - 0.4 * np.clip(slack / (np.abs(slack_30) + eps), 0.0, 1.0), 0.2, 1.0))
    deadline_penalty = neg_slack * uncertainty_mod
    # Hard amplification for deeply urgent tasks
    deadline_penalty = np.where(is_deeply_urgent, deadline_penalty + 15.0, deadline_penalty)

    # Criticality-energy synergy: upward_rank * (duration + comm_weighted) / energy
    # Enhanced with communication-aware latency (Parent 1 coupling)
    comm_weighted_duration = duration + 0.3 * min_comm_time * (1.0 + 0.5 * robust_scale(upward_rank))
    synergy = upward_rank * comm_weighted_duration / (min_incremental_energy + eps)
    synergy_score = -robust_scale(synergy)

    # Energy efficiency gating: only penalize above-median energy density (Parent 1)
    energy_density = np.clip(min_incremental_energy / (duration + eps), eps, 1e6)
    median_energy_den = np.median(energy_density) + eps
    energy_eff_mask = (energy_density <= median_energy_den).astype(float)
    energy_base = min_incremental_energy * (1.0 + 0.6 * np.clip(uncertainty, 0.0, 1.0))
    energy_score = robust_scale(energy_base) * (1.0 - energy_eff_mask)

    # Latency score: duration scaled independently
    latency_score = robust_scale(duration)

    # Starvation mitigation: activated only for non-urgent, non-critical tasks (Parent 1 decay + Parent 2 threshold)
    wait_thresh = np.quantile(ready_wait_time, 0.75) if N > 0 else eps
    wait_boost_raw = np.where(ready_wait_time > wait_thresh, 
                             (ready_wait_time - wait_thresh) / (np.maximum(np.std(ready_wait_time), eps) + eps), 
                             0.0)
    # Decay with criticality: less starvation relief for high-upward-rank tasks
    starv_decay = np.exp(-0.7 * robust_scale(upward_rank))
    wait_score = -wait_boost_raw * starv_decay * np.where(is_deeply_urgent | (upward_rank > np.quantile(upward_rank, 0.8)), 0.0, 1.0)

    # Work penalty: only when urgent, proportional to remaining work and slack deficit (Parent 2 hard gating)
    slack_gap = np.clip(slack_30 - slack, 0.0, None)
    work_norm = robust_scale(remaining_work)
    work_penalty = work_norm * (slack_gap / (np.abs(slack_30) + eps))

    # Final weighted combination enforcing strict dominance hierarchy
    # Urgency dominates; synergy balances speed/importance/efficiency; energy/latency/fairness are refinements
    score = (
        2.0 * deadline_penalty +           # Primary: hard deadline enforcement
        0.5 * synergy_score +              # Secondary: criticality-latency-energy balance
        0.25 * energy_score +              # Tertiary: energy efficiency (gated)
        0.15 * latency_score +             # Quaternary: pure latency minimization
        0.1 * wait_score +                 # Quinary: fairness for low-criticality starved tasks
        0.1 * work_penalty                 # Bonus: subtree workload pressure under urgency
    )

    # Ensure finiteness and shape compliance
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e9, 1e9)
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
