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
    v2 priority rule: Unified urgency-energy coupling + adaptive starvation rescue + robust deadline-proximity normalization.
    
    Key synthesis:
    - Combines Parent 2's unified violation-aware energy exponentiation (smooth, monotonic, no thresholds)
    - Adopts Parent 2's relaxed starvation criteria (delayed high-criticality tasks rescued) but adds criticality gating from Parent 1
    - Integrates Parent 1's robust trimmed-minmax normalization with Parent 2's N=1/2 fallback logic for numerical stability
    - Introduces *criticality-weighted urgency amplification*: urgency penalty scaled by upward_rank to prioritize late tasks on critical path
    - Adds *energy-urgency decoupling*: energy penalty attenuated for highly urgent tasks to avoid over-penalizing feasible critical work
    - Tightens uncertainty handling: dur_uncertainty used both in energy exponent and as direct penalty term
    - Final weights emphasize hard-DDL feasibility (0.60), then synergy (0.22), energy-risk (0.08), comm (0.06), starvation (0.04)
    """
    eps = 1e-08
    min_exec_time = np.asarray(min_exec_time, dtype=float).copy()
    min_comm_time = np.asarray(min_comm_time, dtype=float).copy()
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float).copy()
    slack = np.asarray(slack, dtype=float).copy()
    upward_rank = np.asarray(upward_rank, dtype=float).copy()
    remaining_work = np.asarray(remaining_work, dtype=float).copy()
    ready_wait_time = np.asarray(ready_wait_time, dtype=float).copy()
    uncertainty = np.asarray(uncertainty, dtype=float).copy()
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=float)

    def robust_trimmed_minmax(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.clip(x, -1e12, 1e12)
        if N == 1:
            return np.array([0.0])
        elif N == 2:
            x_sorted = np.sort(x)
            x_min, x_max = x_sorted[0], x_sorted[1]
            rng = x_max - x_min + eps
            return np.clip((x - x_min) / rng, 0.0, 1.0)
        else:
            x_sorted = np.sort(x)
            trim_n = max(1, int(0.1 * len(x_sorted)))
            x_trimmed = x_sorted[trim_n:-trim_n] if len(x_sorted) > 2 * trim_n else x_sorted
            x_min, x_max = np.min(x_trimmed), np.max(x_trimmed)
            rng = x_max - x_min + eps
            return np.clip((x - x_min) / rng, 0.0, 1.0)

    task_duration = np.maximum(min_exec_time + min_comm_time, eps)
    rel_slack = np.divide(slack, task_duration, out=np.zeros_like(slack), where=task_duration != 0)
    rel_slack = np.nan_to_num(rel_slack, nan=0.0, posinf=0.0, neginf=0.0)

    # Criticality-weighted urgency penalty: linear in lateness, scaled by upward_rank
    violated_mask = slack < 0
    tight_mask = ~violated_mask & (rel_slack < 0.5)
    relaxed_mask = ~violated_mask & (rel_slack >= 0.5)
    
    urgency_penalty = np.zeros_like(slack)
    # Violated: strong penalty proportional to lateness normalized by duration
    urgency_penalty[violated_mask] = np.clip(-slack[violated_mask] / (task_duration[violated_mask] + eps), 0.0, 8.0)
    # Tight: moderate penalty based on relative slack distance
    urgency_penalty[tight_mask] = np.clip(0.5 - rel_slack[tight_mask], 0.0, 3.0)
    # Relaxed: weak penalty to preserve fairness
    urgency_penalty[relaxed_mask] = np.clip(1.0 - rel_slack[relaxed_mask], 0.0, 0.5)
    
    # Amplify urgency by criticality: higher upward_rank → stronger penalty for same slack
    urgency_penalty = urgency_penalty * (1.0 + 0.5 * robust_trimmed_minmax(upward_rank))

    # Unified violation-aware energy exponentiation (Parent 2 core)
    rel_slack_distance = np.maximum(0.0, -slack) / (task_duration + eps)
    energy_exponent = 1.3
    risk_weighted_energy = min_incremental_energy * np.power(1.0 + uncertainty * rel_slack_distance + eps, energy_exponent)
    risk_weighted_energy = np.clip(risk_weighted_energy, eps, 1e10)

    # Energy-urgency decoupling: attenuate energy penalty for highly urgent tasks
    norm_urgency = robust_trimmed_minmax(urgency_penalty)
    energy_attenuation = 1.0 - np.clip(norm_urgency, 0.0, 1.0)
    risk_weighted_energy = risk_weighted_energy * (0.3 + 0.7 * energy_attenuation)

    # Synergy term: latency-critical computational synergy (Parent 2 style, enhanced)
    median_task_dur = np.median(task_duration) + eps
    slack_factor = np.clip(1.0 - np.maximum(0.0, slack - 0.5 * median_task_dur) / (task_duration + eps), 0.1, 1.0)
    dampened_ur = upward_rank * slack_factor
    base_synergy = dampened_ur * task_duration / (min_incremental_energy + eps)
    synergy_amplifier = np.where(violated_mask, 1.0 + 0.8 * uncertainty, 1.0)
    latency_crit_synergy = base_synergy * synergy_amplifier
    latency_crit_synergy = np.clip(latency_crit_synergy, eps, 1e9)

    # Communication pressure: uncertainty-weighted + slack-distance scaled (Parent 2)
    comm_overhead_ratio = min_comm_time / (task_duration + eps)
    comm_pressure = comm_overhead_ratio * (1.0 + 0.6 * uncertainty)
    slack_distance_penalty = np.clip(1.0 + np.maximum(0.0, -slack) / (median_task_dur), 1.0, 4.0)
    comm_pressure = np.where(violated_mask, comm_pressure * 3.0 * slack_distance_penalty,
                            np.where(tight_mask, comm_pressure * 1.5 * slack_distance_penalty,
                                    comm_pressure * 0.5))

    # Starvation rescue: relaxed criteria (Parent 2) + criticality gating (Parent 1)
    median_rw = np.median(remaining_work) + eps
    rw_normalized = np.clip(remaining_work / median_rw, 0.01, 100.0)
    norm_wait_ratio = np.clip(ready_wait_time / median_task_dur, 0.0, 10.0)
    is_starvable = (norm_wait_ratio > 0.7) & (rw_normalized > 0.2) & (slack < 180.0)
    # Prioritize starvable tasks with high upward_rank (critical path awareness from Parent 1)
    starvability_score = np.where(is_starvable, 
                                 norm_wait_ratio * (1.0 + 0.3 * robust_trimmed_minmax(upward_rank)), 
                                 0.0)
    starvation_boost = np.clip(starvability_score, 0.0, 1.5)

    # Uncertainty penalty: direct contribution for high-risk tasks
    dur_uncertainty = np.divide(uncertainty, task_duration + eps, out=np.zeros_like(uncertainty), where=task_duration + eps != 0)
    dur_uncertainty = np.nan_to_num(dur_uncertainty, nan=0.0, posinf=0.0, neginf=0.0)
    uncertainty_penalty = robust_trimmed_minmax(dur_uncertainty) * 0.8

    # Normalize all components
    norm_urgency = robust_trimmed_minmax(urgency_penalty)
    norm_synergy = robust_trimmed_minmax(latency_crit_synergy)
    norm_energy = robust_trimmed_minmax(risk_weighted_energy)
    norm_comm = robust_trimmed_minmax(comm_pressure)
    norm_starvation = robust_trimmed_minmax(starvation_boost)
    norm_uncertainty = robust_trimmed_minmax(dur_uncertainty)

    # Weighted score: smaller = better
    score = (
        0.60 * norm_urgency +           # Hard-DDL feasibility dominates
        -0.22 * norm_synergy +          # Positive synergy (higher = better → negative weight)
        0.08 * norm_energy +            # Risk-adjusted energy penalty
        0.06 * norm_comm +              # Communication overhead penalty
        0.04 * norm_starvation +        # Starvation relief boost (lower score when active)
        0.03 * norm_uncertainty         # Direct uncertainty penalty
    )

    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    score = np.clip(score, -1e12, 1e12)
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
