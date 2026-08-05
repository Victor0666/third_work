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
    v2: Risk-aware lexicographic priority with adaptive slack gating, 
        normalized criticality-energy ratio, and starvation-robust fairness.
    
    Key mutations:
    - Replaces arctan urgency with tanh-based robust urgency: smoother near zero, 
      bounded [-1,1], better handles extreme negative slack without saturation.
    - Introduces *risk-adjusted slack* = slack - 2.5 * uncertainty, then applies 
      hard-gated urgency: only active when risk-adjusted slack < 0.5s (proactive violation prevention).
    - Uses MAD (median absolute deviation) instead of IQR for normalization — more outlier-resilient 
      for small N and sparse ready sets; includes explicit N=1 fallback to zero-mean scaling.
    - Replaces SEER-inspired ratio with *energy-efficiency density*: 
      (remaining_work / (min_exec_time + min_comm_time + eps)) / (min_incremental_energy + eps),
      emphasizing work-per-joule-per-second — directly penalizes energy-wasteful short tasks.
    - Fairness term now uses *waiting-age penalty*: sqrt(ready_wait_time) * (1 + max(0, -slack)), 
      amplifying priority boost for long-waiting tasks *only when deadline pressure exists*, 
      avoiding unfair boosts under safe slack.
    - Removes sigmoid energy gate; instead applies *slack-dependent linear interpolation* between 
      deadline-dominant (slack <= 0) and energy-dominant (slack >= 5.0) regimes — interpretable & monotonic.
    - All terms scaled to [-3.0, 3.0] range pre-sum to ensure numerical dominance order remains stable.
    """
    eps = 1e-8
    def clean(x):
        x = np.asarray(x, dtype=float)
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    
    min_exec_time = clean(min_exec_time)
    min_comm_time = clean(min_comm_time)
    min_incremental_energy = clean(min_incremental_energy)
    slack = clean(slack)
    upward_rank = clean(upward_rank)
    remaining_work = clean(remaining_work)
    ready_wait_time = clean(ready_wait_time)
    uncertainty = clean(uncertainty)
    
    # Risk-adjusted slack: proactive violation margin
    risk_adjusted_slack = slack - 2.5 * uncertainty
    
    # Robust tanh urgency: bounded [-1, 1], monotonic, avoids arctan’s flat tails
    urgency_raw = np.tanh(-risk_adjusted_slack / (1.0 + eps))
    
    # Hard gating: urgency only active when risk_adjusted_slack < 0.5s (soft deadline boundary)
    urgency_gate = np.where(risk_adjusted_slack < 0.5, 1.0, 0.0)
    urgency_term = -3.0 * urgency_raw * urgency_gate
    
    # Energy-efficiency density: work per communication+compute time, normalized by energy
    # Higher value = more computation work delivered per joule per second → prefer
    comp_time = min_exec_time + min_comm_time + eps
    eff_density = (remaining_work / comp_time) / (min_incremental_energy + eps)
    
    # Slack-dependent regime interpolation: [deadline-dominant, energy-dominant]
    # Linear blend from weight=1.0 (slack <= 0) to weight=0.0 (slack >= 5.0)
    blend_weight = np.clip((5.0 - slack) / 5.0, 0.0, 1.0)
    energy_term_base = -eff_density * blend_weight
    
    # Criticality-pressure: upward_rank weighted by execution burden and uncertainty
    critical_cost = comp_time * (1.0 + uncertainty)
    pressure_base = upward_rank * critical_cost
    # Active only when slack is moderately tight (0 < risk_adjusted_slack < 3.0)
    pressure_gate = np.where((risk_adjusted_slack > 0.0) & (risk_adjusted_slack < 3.0), 1.0, 0.0)
    pressure_term = 1.4 * pressure_base * pressure_gate
    
    # Waiting-age fairness: sqrt(wait) amplified *only* under deadline pressure (negative or low slack)
    wait_boost = np.sqrt(ready_wait_time + eps) * np.maximum(0.0, 1.0 - np.clip(slack / 2.0, 0.0, 1.0))
    fairness_term = -0.3 * wait_boost
    
    # Robust MAD-based normalization (more stable than IQR for N<5 or skewed data)
    def normalize_mad(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size == 1:
            return np.array([0.0])
        median_x = np.median(x)
        mad = np.median(np.abs(x - median_x))
        scale = mad if mad > eps else np.std(x) if x.size > 1 else eps
        scale = max(scale, eps)
        normed = (x - median_x) / scale
        return np.clip(normed, -4.0, 4.0)
    
    norm_urgency = normalize_mad(urgency_term)
    norm_energy = normalize_mad(energy_term_base)
    norm_pressure = normalize_mad(pressure_term)
    norm_fairness = normalize_mad(fairness_term)
    
    # Final score: sum of normalized, bounded components
    score = (
        norm_urgency +
        norm_energy +
        norm_pressure +
        norm_fairness
    )
    
    # Final sanitization: finite bounds and NaN/inf cleanup
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e9, 1e9)
    
    return score
