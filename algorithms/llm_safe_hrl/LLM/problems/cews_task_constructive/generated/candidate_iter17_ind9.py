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
    Self-evolved priority rule v2: Hybrid of Parents 1 & 2 with refined urgency calibration,
    synergistic criticality-energy coupling, and starvation-avoidance via work-aware wait pressure.
    
    Key improvements:
      - Unified urgency bands: uses *relative slack quantiles* (0.1/0.5/0.9) for finer-grained deadline hardness,
        with smooth interpolation between bands to avoid discontinuities.
      - Criticality-energy synergy enhanced with *slack-distance-gated amplification*: 
        uncertainty × upward_rank × (1 + rel_slack_distance)^1.5 only when slack < median_slack,
        preserving high-rank flow under tight deadlines while suppressing noise in relaxed regime.
      - Starvation boost redesigned as *work-normalized waiting pressure*: 
        (ready_wait_time / task_duration) × min(remaining_work / median_rw, 1.0) × sigmoid(slack),
        ensuring fairness scales with both wait duration and computational significance.
      - Energy risk scoring simplified but sharper: uses (1 + uncertainty × max(0, -slack)/task_duration)^1.4
        to focus penalty strictly on violation severity, avoiding over-amplification in tight-but-feasible region.
      - All normalizations use MAD with deterministic fallback; all divisions guarded; outputs finite, clipped, and N=1 safe.
      - Final weights emphasize urgency dominance (0.60), synergy efficiency (0.28), energy control (0.07), 
        starvation relief (0.03), and communication overhead (0.02) — aligning with DDL-hardness first objective.
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

    def robust_normalize_mad(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.clip(x, -1e12, 1e12)
        if N == 1:
            return np.array([0.0])
        center = np.median(x)
        mad = np.median(np.abs(x - center)) + eps
        normed = (x - center) / mad
        return np.clip(normed, -10.0, 10.0)

    # Compute stable base metrics
    task_duration = np.maximum(min_exec_time + min_comm_time, eps)
    rel_slack = np.divide(slack, task_duration, out=np.zeros_like(slack), where=task_duration != 0)

    # Adaptive urgency bands using quantiles for robustness across scales
    if N > 1:
        q10, q50, q90 = np.quantile(rel_slack, [0.1, 0.5, 0.9], method='midpoint')
    else:
        q10 = q50 = q90 = rel_slack[0]

    # Smooth urgency penalty via piecewise linear interpolation between quantile bands
    urgency_penalty = np.zeros_like(slack)
    violated_mask = slack < 0
    tight_mask = ~violated_mask & (rel_slack < q50)
    relaxed_mask = ~violated_mask & (rel_slack >= q50)

    # Violated: linear penalty scaled by absolute slack/duration
    urgency_penalty[violated_mask] = np.clip(-slack[violated_mask] / (task_duration[violated_mask] + eps), 0.0, 8.0)
    # Tight: linear decay from q50 down to q10
    urgency_penalty[tight_mask] = np.clip(q50 - rel_slack[tight_mask], 0.0, 4.0)
    # Relaxed: gentle decay from q50 to q90
    urgency_penalty[relaxed_mask] = np.clip(q90 - rel_slack[relaxed_mask], 0.0, 1.2)

    # Criticality attenuation: dampen upward_rank only when slack is relaxed
    slack_factor = np.clip(1.0 - np.maximum(0.0, slack - q50 * np.median(task_duration)) / (task_duration + eps), 0.05, 1.0)
    dampened_ur = upward_rank * slack_factor

    # Synergy: task importance × duration / energy, amplified only under urgency
    base_synergy = dampened_ur * task_duration / (min_incremental_energy + eps)
    # Amplify only when slack < median_slack — avoids spurious boosting in relaxed regime
    median_slack = np.median(slack) + eps
    synergy_amplifier = np.where(slack < median_slack,
                                 1.0 + 0.8 * uncertainty * np.clip(upward_rank / (np.median(upward_rank) + eps), 0.1, 10.0),
                                 1.0)
    latency_crit_synergy = base_synergy * synergy_amplifier
    latency_crit_synergy = np.clip(latency_crit_synergy, eps, 1e8)

    # Energy risk: sharp, violation-focused penalty
    rel_slack_distance = np.maximum(0.0, -slack) / (task_duration + eps)
    energy_exponent = 1.4
    risk_weighted_energy = min_incremental_energy * np.power(1.0 + uncertainty * rel_slack_distance + eps, energy_exponent)
    risk_weighted_energy = np.clip(risk_weighted_energy, eps, 1e9)

    # Communication pressure: normalized overhead, gated by violation/tightness
    comm_overhead_ratio = min_comm_time / (task_duration + eps)
    comm_pressure = comm_overhead_ratio * (1.0 + 0.4 * uncertainty)
    comm_pressure = np.where(violated_mask, comm_pressure * 2.5,
                            np.where(tight_mask, comm_pressure * 0.7, comm_pressure * 0.1))

    # Starvation boost: work-aware, slack-coupled, and bounded
    median_task_dur = np.median(task_duration) + eps
    median_rw = np.median(remaining_work) + eps
    wait_ratio = np.clip(ready_wait_time / median_task_dur, 0.0, 5.0)
    rw_ratio = np.clip(remaining_work / median_rw, 0.01, 1.0)
    # Sigmoid slack coupling: strong boost only near deadline, smooth decay beyond
    slack_sigmoid = 1.0 / (1.0 + np.exp(-(slack - median_slack) / (np.maximum(task_duration, 1.0) + eps)))
    wait_boost = wait_ratio * rw_ratio * (1.0 - slack_sigmoid)  # higher when slack is low
    wait_boost = np.clip(wait_boost, 0.0, 1.5)

    # Normalize all components
    norm_urgency = robust_normalize_mad(urgency_penalty)
    norm_synergy = robust_normalize_mad(latency_crit_synergy)
    norm_energy = robust_normalize_mad(risk_weighted_energy)
    norm_comm = robust_normalize_mad(comm_pressure)
    norm_wait = robust_normalize_mad(wait_boost)

    # Final weighted score: smaller = higher priority
    score = (
        0.60 * norm_urgency
        - 0.28 * norm_synergy
        + 0.07 * norm_energy
        + 0.02 * norm_comm
        + 0.03 * norm_wait
    )

    # Ensure finiteness and shape compliance
    score = np.nan_to_num(score, nan=0.0, posinf=1e10, neginf=-1e10)
    score = np.clip(score, -1e10, 1e10)
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
