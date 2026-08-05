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
    Hybrid priority rule v2: Deadline-dominant + risk-decoupled synergy + fairness-gated starvation.
    
    Key hybrid improvements:
    - Uses Parent 2's strict deadline dominance and decoupled uncertainty modulation (only on lateness)
    - Adopts Parent 1's adaptive urgency steepness (eta) for sharper near-deadline response
    - Integrates Parent 2's robust_scale with Parent 1's critical-path fidelity term for stability
    - Introduces work-normalized starvation gating: activates only when (wait_ratio > p90) AND (slack >= 0)
    - Replaces energy density ratio with harmonic efficiency term from Parent 1, but gated by slack >= 0
    - All operations protected with eps; no NaN/inf; deterministic; shape-(N,) guaranteed.
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

    def robust_scale(x):
        if len(x) == 0:
            return x
        med = np.median(x)
        x_centered = x - med
        q1, q3 = np.quantile(x_centered, [0.25, 0.75], method='higher')
        iqr = q3 - q1
        if iqr < eps:
            mad = np.median(np.abs(x_centered))
            scale = mad if mad > eps else np.mean(np.abs(x_centered)) + eps
        else:
            scale = iqr + eps
        scaled = x_centered / scale
        return np.clip(scaled, -1000000.0, 1000000.0)

    # Task duration and adaptive urgency steepness (Parent 1)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    task_min_duration = np.maximum(min_exec_time + min_comm_time, eps)
    norm_lateness_abs = np.abs(slack) / (task_min_duration + eps)
    eta = np.clip(4.0 + 2.0 * (1.0 - np.exp(-norm_lateness_abs)), 2.0, 6.0)
    # Deadline urgency: sigmoid with adaptive steepness, inverted so smaller = more urgent
    deadline_urgency = 1.0 / (1.0 + np.exp(-eta * -slack / (task_min_duration + eps)))
    
    # Lateness penalty: decoupled uncertainty modulation (Parent 2)
    neg_slack = np.maximum(-slack, 0.0)
    uncertainty_mod = 1.0 + 0.9 * np.clip(uncertainty, 0.0, 2.0)
    deadline_penalty = neg_slack * uncertainty_mod
    
    # Critical-path fidelity: deviation from median upward_rank (Parent 1)
    cp_fidelity = np.abs(upward_rank - np.median(upward_rank))
    q1_ur, q3_ur = np.quantile(upward_rank, [0.25, 0.75], method='higher')
    iqr_ur = q3_ur - q1_ur + eps
    cp_fidelity_norm = robust_scale(cp_fidelity / iqr_ur)
    
    # Synergy term: upward_rank * normalized duration / energy (Parent 2)
    duration_med = np.median(duration) + eps
    norm_duration = duration / duration_med
    synergy_denom = min_incremental_energy + eps
    synergy = upward_rank * norm_duration / synergy_denom
    synergy_score = -robust_scale(synergy)
    
    # Harmonic efficiency: energy-work balance, activated only when slack >= 0 (hybrid)
    energy_per_work = min_incremental_energy / (remaining_work + eps)
    work_efficiency_ratio = remaining_work / (min_incremental_energy + eps)
    harmonic_eff = 2.0 / (1.0 / (energy_per_work + eps) + 1.0 / (work_efficiency_ratio + eps))
    harmonic_eff = harmonic_eff * (1.0 + 0.3 * np.clip(uncertainty, 0.0, 1.0))
    harmonic_eff = np.clip(harmonic_eff, 1e-06, 1e6)
    norm_energy_eff = robust_scale(harmonic_eff)
    # Gate harmonic term by slack feasibility (avoids penalizing late tasks unfairly)
    harmonic_gate = (slack >= 0.0).astype(float)
    harmonic_score = norm_energy_eff * harmonic_gate
    
    # Starvation boost: work-normalized and slack-gated (hybrid)
    wait_per_work = ready_wait_time / (remaining_work + eps)
    p90_wait_ratio = np.quantile(wait_per_work, 90, method='higher') if N > 1 else np.max(wait_per_work)
    is_starvable = (wait_per_work > p90_wait_ratio + eps) & (slack >= 0.0)
    starvation_boost = np.where(is_starvable, robust_scale(ready_wait_time) * robust_scale(wait_per_work), 0.0)
    
    # Work and latency normalization
    work_norm = robust_scale(remaining_work)
    latency_score = robust_scale(duration)
    
    # Final weighted score: smaller = higher priority
    # Weights chosen to emphasize deadline penalty and synergy while balancing fairness
    score = (
        3.0 * deadline_penalty +
        0.8 * synergy_score +
        0.4 * harmonic_score +
        0.15 * latency_score +
        0.1 * work_norm +
        0.25 * cp_fidelity_norm +
        0.2 * starvation_boost -
        2.5 * deadline_urgency  # Urgency term subtracted to make small values high priority
    )
    
    # Robust finalization
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    score = np.clip(score, -1e12, 1e12)
    
    # Ensure shape is (N,)
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    
    return score
