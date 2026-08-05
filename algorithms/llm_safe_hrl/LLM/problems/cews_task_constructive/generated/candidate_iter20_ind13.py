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
    v2 synthesis: Combines tanh urgency & multiplicative structure (v1) with criticality-pressure
    and robust fairness (v0), while enhancing deadline safety, energy-gating, and numerical stability.
    
    Key improvements:
    - Uses tanh(-slack/τ) urgency (v1) but with τ=0.3 for sharper violation response.
    - Critical-path density (CPD) enhanced with communication-aware criticality: upward_rank / (min_exec_time + min_comm_time + ε).
    - Energy-latency efficiency (ELER) replaced by *CED-gated energy efficiency*: 
      (upward_rank * remaining_work) / (min_incremental_energy + ε), activated only when slack > 1.0s (tighter than v1's 2.0s, looser than v0's 0.5s).
    - Introduces *latency-pressure term*: (min_exec_time + min_comm_time) * (1 + uncertainty) * (1 - sigmoid(slack + 0.1)), 
      active only near/after deadline, normalized and bounded to prevent dominance skew.
    - Fairness uses sqrt(wait) * exp(-uncertainty) * (1 + |slack|)^-0.5 — always active but attenuated under tight deadlines.
    - All normalizations use MAD + clipping [-2.5, 2.5] for outlier resilience and N=1 stability.
    - Final score = urgency × (1 + CPD_norm) × (1 + CED_norm) + pressure_term + fairness_term, 
      with all components clipped pre-composition and final score bounded.
    """
    eps = 1e-08
    # Safe casting and NaN/inf handling
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

    def normalize_mad_clipped(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size == 1:
            return np.array([0.0])
        median_x = np.median(x)
        abs_devs = np.abs(x - median_x)
        mad = np.median(abs_devs)
        scale = mad if mad > eps else eps
        z = (x - median_x) / scale
        return np.clip(z, -2.5, 2.5)

    # Urgency: tanh-based, sharper deadline sensitivity (τ=0.3)
    tau_urgency = 0.3
    urgency_raw = np.tanh(-slack / tau_urgency)
    norm_urgency = normalize_mad_clipped(urgency_raw)
    urgency_term = 1.0 + 0.9 * norm_urgency  # stronger urgency weight than v1

    # Critical-path density: communication-aware (exec + comm in denominator)
    cpd_base = upward_rank / (min_exec_time + min_comm_time + eps)
    norm_cpd = normalize_mad_clipped(cpd_base)
    cpd_term = 1.0 + 0.7 * norm_cpd

    # CED-gated energy efficiency: (importance × work) / energy, enabled only when slack > 1.0s
    ced_base = (upward_rank * remaining_work) / (min_incremental_energy + eps)
    ced_mask = (slack > 1.0).astype(float)
    ced_gated = ced_base * ced_mask
    norm_ced = normalize_mad_clipped(ced_gated)
    ced_term = 1.0 + 0.5 * norm_ced

    # Latency-pressure term: active only when slack <= 0.1s (near-violation or violated)
    # Sigmoid gating: soft transition around slack = -0.1s → 1 - sigmoid(slack + 0.1)
    pressure_gate = 1.0 - 1.0 / (1.0 + np.exp(slack + 0.1))  # ≈ 0 when slack >> 0, ≈ 1 when slack << 0
    latency_pressure_base = (min_exec_time + min_comm_time) * (1.0 + uncertainty)
    pressure_raw = latency_pressure_base * pressure_gate
    norm_pressure = normalize_mad_clipped(pressure_raw)
    pressure_term = 0.8 * norm_pressure  # additive, not multiplicative — avoids explosion

    # Fairness: starvation prevention with uncertainty attenuation and slack-based decay
    wait_safe = np.maximum(ready_wait_time, 0.0)
    sqrt_wait = np.sqrt(wait_safe + eps)
    exp_uncert = np.exp(-np.clip(uncertainty, 0.0, 10.0))
    # Decay fairness under tight deadlines: (1 + |slack|)^-0.5 → larger penalty when slack is large positive, less decay when negative
    slack_decay = (1.0 + np.abs(slack)) ** (-0.5)
    fairness_raw = sqrt_wait * exp_uncert * slack_decay
    norm_fairness = normalize_mad_clipped(fairness_raw)
    fairness_term = -0.2 * norm_fairness

    # Multiplicative core + additive corrections
    core_score = urgency_term * cpd_term * ced_term
    score = core_score + pressure_term + fairness_term

    # Final sanitization
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e8, 1e8)

    return score
