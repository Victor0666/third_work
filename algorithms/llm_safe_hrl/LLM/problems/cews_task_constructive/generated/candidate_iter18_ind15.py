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
    v2 self-evolved priority rule: Streamlined, deadline-hardened, and globally scaled.
    - Replaces per-term robust normalization with *single global context-aware scaling*:
      uses MAD over all urgency-relevant base terms to preserve relative hierarchy in small-N sets.
    - Unifies energy awareness: `energy_efficiency = 1/(min_incremental_energy + ε)` is *always active*,
      but weighted by smooth slack-gating `sigmoid(slack/τ)` — enabling aggressive efficiency when safe,
      graceful de-prioritization near deadlines (no hard gates → avoids priority cliffs).
    - Eliminates redundant latency-risk ratio; instead strengthens CPD (`upward_rank / exec`) with
      explicit communication-awareness: `CPD_comm = upward_rank / (min_exec_time + min_comm_time + ε)`.
    - Urgency uses clipped tanh(-slack/τ) with *global normalization* (not per-term), preserving sharp
      deadline boundary response without numerical fragility.
    - Fairness is simplified: `sqrt(ready_wait_time)` scaled by `exp(-uncertainty)` *only when slack > 0`,
      capped and globally normalized — prevents starvation while respecting DDL safety.
    - All terms are finite, zero/Nan/inf protected, deterministic, and return shape-(N,) array.
    - Final score is linear combination with coefficients tuned to prioritize deadline compliance first,
      then energy, then fairness — aligned with "DDL-hard + energy-minimizing" objective.
    """
    eps = 1e-08
    N = len(np.atleast_1d(min_exec_time))
    
    # Safe casting & nan/inf handling
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=eps, posinf=eps, neginf=eps)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=1e6, neginf=-1e6)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=eps, posinf=eps, neginf=eps)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    
    # Global context for unified scaling: collect all base signals influencing priority
    # (urgency, criticality, energy-efficiency, fairness — no latency-risk duplication)
    base_signals = []
    
    # 1. Urgency: tanh-based, clipped for stability, unnormalized raw
    tau_urg = 0.5
    urgency_raw = np.tanh(-np.clip(slack, -100.0, 100.0) / tau_urg)
    base_signals.append(urgency_raw)
    
    # 2. Critical-path density (communication-aware): higher rank / lower latency → high priority
    cpd_comm = upward_rank / (min_exec_time + min_comm_time + eps)
    base_signals.append(cpd_comm)
    
    # 3. Energy efficiency: always active, smoothly gated by slack safety
    energy_efficiency = 1.0 / (min_incremental_energy + eps)
    slack_gate = 1.0 / (1.0 + np.exp(-slack / 1.0))  # sigmoid: ~0 when slack << 0, ~1 when slack >> 0
    seer_gated = energy_efficiency * slack_gate
    base_signals.append(seer_gated)
    
    # 4. Fairness: sqrt(wait) scaled only under safe slack, decayed by uncertainty
    wait_safe = np.maximum(ready_wait_time, 0.0)
    sqrt_wait = np.sqrt(wait_safe + eps)
    fairness_raw = np.where(slack > 0.0, sqrt_wait * np.exp(-uncertainty), 0.0)
    fairness_clipped = np.clip(fairness_raw, 0.0, 0.25)
    base_signals.append(fairness_clipped)
    
    # Global robust scaling: compute MAD over concatenated base signals for stable normalization
    if N == 1:
        norm_scale = eps
        norm_shift = 0.0
    else:
        all_base = np.concatenate([s.reshape(-1) for s in base_signals])
        med = np.median(all_base)
        abs_dev = np.abs(all_base - med)
        mad = np.median(abs_dev)
        norm_scale = mad if mad > eps else (np.max(all_base) - np.min(all_base) + eps)
        norm_shift = med
    
    def global_normalize(x):
        return (x - norm_shift) / (norm_scale + eps)
    
    # Apply global normalization to each term
    norm_urgency = global_normalize(urgency_raw)
    norm_cpd = global_normalize(cpd_comm)
    norm_seer = global_normalize(seer_gated)
    norm_fairness = global_normalize(fairness_clipped)
    
    # Weighted combination: deadline-first (urgency + cpd), then energy, then fairness
    # Coefficients sum to ~1.0 and reflect objective hierarchy: DDL hardness > energy > fairness
    urgency_term = -4.0 * norm_urgency      # strongest weight: deadline compliance is primary
    cpd_term = -2.5 * norm_cpd             # critical path drives scheduling order under tight slack
    seer_term = -1.8 * norm_seer           # energy efficiency promoted when safe
    fairness_term = -0.7 * norm_fairness   # mild starvation prevention, only when slack permits
    
    # Final score: smaller = higher priority
    score = urgency_term + cpd_term + seer_term + fairness_term
    
    # Ensure finite output
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    
    # Enforce shape (N,)
    return score.reshape(-1)
