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
    Hybrid priority rule v2: Adaptive deadline urgency + risk-aware criticality + robust energy fairness + gated starvation mitigation.
    
    Key innovations:
    - Combines Parent 2's smooth tanh urgency cliff with Parent 1's lexicographic intent via weighted dominance (urgency > criticality > energy)
    - Introduces *risk-gated criticality*: upward_rank scaled by uncertainty-weighted slack penalty, not just raw uncertainty
    - Uses *work-normalized energy density* with MAD-robust scaling and explicit zero-safety for small work
    - Starvation boost activated only when (slack > 0) AND (uncertainty < median) AND (wait_ratio > 90th percentile), ensuring fairness without compromising deadlines
    - All normalizations use trimmed-median centering + MAD for stability on small N; all divisions guarded; all outputs finite and shape-correct
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

    # Robust normalization: trimmed median center + MAD scaling
    def robust_mad_normalize(x):
        if x.size == 0:
            return np.zeros_like(x)
        x = x.astype(float)
        # Use median for center when N > 3, else mean for better small-N bias
        center = np.median(x) if N > 3 else np.mean(x)
        abs_devs = np.abs(x - center)
        mad = np.median(abs_devs) + eps
        normed = (x - center) / mad
        return np.clip(normed, -8.0, 8.0)

    # --- Deadline Urgency (Parent 2 style, enhanced) ---
    task_duration = np.maximum(min_exec_time + min_comm_time, eps)
    urgency_base = -slack / task_duration
    urgency_cliff = np.tanh(urgency_base * 0.8)  # Smooth saturation at ±1
    # Binary hard-deadline penalty amplified only when slack <= 0
    deadline_penalty = np.where(slack <= 0, 1.0, 0.0)
    deadline_urgency = 1.0 + 0.95 * np.maximum(0.0, urgency_cliff) + 0.05 * deadline_penalty
    norm_urgency = robust_mad_normalize(deadline_urgency)

    # --- Risk-Aware Criticality (Hybrid: Parent 2's risk factor + Parent 1's cp_leverage intent) ---
    # Risk factor: uncertainty penalizes critical tasks more under lateness pressure
    risk_factor = np.clip(1.0 + uncertainty * np.maximum(0.0, -slack), 1.0, 5.0)
    # Criticality per unit energy: upward_rank normalized by risk-adjusted energy cost
    crit_score = upward_rank / (min_incremental_energy * risk_factor + eps)
    crit_score = np.clip(crit_score, 1e-6, 1e6)
    norm_crit = robust_mad_normalize(crit_score)

    # --- Energy Fairness (Parent 2 + Parent 1 safety) ---
    work_density = np.maximum(remaining_work, eps)
    energy_per_work = min_incremental_energy / work_density
    energy_per_work = np.clip(energy_per_work, eps, 1e9)
    norm_energy = robust_mad_normalize(energy_per_work)

    # --- Gated Starvation Mitigation (Strictly safer than both parents) ---
    # Eligibility: slack > 0 AND low uncertainty AND high relative wait
    median_unc = np.median(uncertainty) if N > 0 else 0.0
    wait_ratio = np.divide(ready_wait_time, work_density, out=np.zeros_like(ready_wait_time), where=work_density != 0)
    p90_wait = np.percentile(wait_ratio, 90, method='midpoint') if N >= 2 else np.max(wait_ratio)
    starvation_eligible = (slack > 0.0) & (uncertainty < median_unc + eps) & (wait_ratio > p90_wait - eps)
    wait_boost = np.where(starvation_eligible, np.clip(wait_ratio / (p90_wait + eps), 0.0, 0.25), 0.0)

    # --- Weighted fusion: urgency dominates, then criticality, then energy, then fairness ---
    # Lower score = higher priority → negate urgency (since high urgency should be high priority)
    # So: -urgency_term + ... ensures smaller final score for urgent tasks
    score = (
        -2.5 * norm_urgency  # Strongest weight: deadline compliance first
        - 1.5 * norm_crit    # Second: maximize critical-path progress per energy
        + 0.4 * norm_energy  # Third: penalize high energy-per-work, but lightly
        + 0.1 * wait_boost   # Fourth: mild fairness boost only when safe
    )

    # Final sanitization: ensure finite, bounded, correct shape
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    score = np.clip(score, -1e12, 1e12)
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
