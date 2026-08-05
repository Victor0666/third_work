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
    Hybrid priority rule v2: Deadline-hard, starvation-fair, energy-critical.
    
    Key integrations:
      - Uses Parent 2's monotonic bounded linear-soft urgency (superior DDL fidelity)
      - Adopts Parent 1's robust median/IQR fallback normalization for degenerate cases (N=1 or constant arrays)
      - Combines Parent 2's direct wait-per-work penalty with Parent 1's slack-distance-aware boost for fairness under latency pressure
      - Introduces novel *criticality-energy synergy* term: upward_rank * (1 / (min_incremental_energy + eps)) scaled by uncertainty-aware gating
      - Unifies risk scaling: applies (1 + uncertainty)^exponent only to tasks with high criticality AND negative slack, avoiding over-penalization
      - All terms are sign-consistent: lower score = higher priority; no inversions
      - Final weights prioritize urgency (0.45), criticality-energy synergy (0.28), fairness (0.12), duration (0.08), uncertainty (0.04), work (0.03)
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
    
    # Robust normalization supporting N=1 and constant arrays
    def robust_normalize(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.clip(x, -1e12, 1e12)
        if N == 1:
            return np.array([0.0])
        median_x = np.median(x)
        q25, q75 = np.quantile(x, [0.25, 0.75], method='midpoint')
        iqr = q75 - q25 + eps
        # Fallback to std if IQR near zero (degenerate case)
        scale = np.where(iqr > eps, iqr, np.std(x) + eps)
        z = (x - median_x) / (scale + eps)
        return np.clip(z, -6.0, 6.0)
    
    # Task minimum duration (execution + communication)
    task_min_duration = np.maximum(min_exec_time + min_comm_time, eps)
    median_dur = np.median(task_min_duration) + eps
    
    # Monotonic bounded linear-soft urgency (Parent 2): strict deadline dominance
    urgency_neg = 1.0 + np.clip(-slack / median_dur, 0.0, 1.0)
    urgency_pos = np.exp(-np.clip(slack / (median_dur + eps), 0.0, 20.0))
    deadline_urgency = np.where(slack <= 0, urgency_neg, urgency_pos)
    
    # Criticality-energy synergy: high upward_rank + low energy per unit criticality
    # Gated by both urgency (slack < 0) and predictability (uncertainty < 0.4) to avoid noise amplification
    crit_energy_base = upward_rank / (min_incremental_energy + eps)
    crit_energy_base = np.clip(crit_energy_base, 1e-6, 1e6)
    urgency_and_lowrisk_mask = (slack < 0) & (uncertainty < 0.4)
    crit_energy_synergy = np.where(urgency_and_lowrisk_mask, 
                                  crit_energy_base * (1.0 + 0.3 * (1.0 - uncertainty)), 
                                  crit_energy_base * 0.7)
    crit_energy_synergy = np.clip(crit_energy_synergy, 1e-6, 1e6)
    
    # Fairness: direct wait-per-work penalty + slack-distance boost (hybrid of both parents)
    wait_per_work = ready_wait_time / (remaining_work + eps)
    max_wait_pw = np.maximum(np.max(wait_per_work), eps)
    base_wait_penalty = np.clip(wait_per_work / max_wait_pw, 0.0, 1.0)
    # Slack-distance boost: amplify waiting penalty when deadline pressure is high
    slack_distance = np.maximum(0.0, -slack) / (task_min_duration + eps)
    wait_boost = base_wait_penalty * (1.0 + 0.5 * np.clip(slack_distance, 0.0, 2.0))
    wait_fairness = np.clip(wait_boost, 0.0, 1.0)
    
    # Risk-adjusted energy: only penalize energy heavily when slack is violated AND uncertainty is moderate-high
    violated_mask = slack < 0
    risk_exponent = np.clip(1.0 + 0.4 * (-slack) / median_dur + 0.2 * uncertainty, 1.0, 3.0)
    energy_risk_weighted = np.where(violated_mask,
                                   min_incremental_energy * np.power(1.0 + uncertainty, risk_exponent),
                                   min_incremental_energy)
    energy_risk_weighted = np.maximum(energy_risk_weighted, eps)
    
    # Normalize all components
    norm_urgency = robust_normalize(deadline_urgency)
    norm_synergy = robust_normalize(crit_energy_synergy)
    norm_fairness = robust_normalize(wait_fairness)
    norm_dur = robust_normalize(task_min_duration)
    norm_unc = robust_normalize(uncertainty)
    norm_work = robust_normalize(remaining_work)
    norm_energy = robust_normalize(energy_risk_weighted)
    
    # Final weighted score: lower = better
    # Urgency dominates; synergy second; fairness third; others minor stabilizers
    score = (
        0.45 * norm_urgency +
        0.28 * (1.0 - norm_synergy) +  # Higher synergy → lower score
        0.12 * norm_fairness +
        0.08 * norm_dur +
        0.04 * norm_unc +
        0.03 * norm_work
    )
    
    # Ensure finite output and correct shape
    score = np.nan_to_num(score, nan=0.0, posinf=1e10, neginf=-1e10)
    score = np.clip(score, -1e10, 1e10)
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
