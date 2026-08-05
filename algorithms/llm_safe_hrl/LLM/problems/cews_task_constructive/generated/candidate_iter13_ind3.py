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
    v2 mutation: Replaces adaptive normalization with robust MAD-based scaling;
    introduces *deadline-feasibility-gated criticality-energy ratio* (CF-CER) 
    instead of CED, using slack-sigmoid gating for smooth transition;
    replaces sqrt-fairness with tanh-based aging that activates only under safe slack;
    adds uncertainty-weighted communication penalty only for edge-bound tasks (implied by low energy);
    uses lexicographic weighting via slack-dependent convex multipliers instead of fixed coefficients;
    eliminates all arctan/pi scaling in favor of numerically stable sigmoid and clipped linear terms.
    
    Key innovations:
    - CF-CER = (upward_rank * remaining_work) / (min_incremental_energy + eps), gated by sigmoid(slack/τ) → smooth [0,1] feasibility mask
    - Latency term splits: exec dominates under urgency (slack < 0), comm dominates under uncertainty + energy-efficiency regime (low min_incremental_energy)
    - Fairness: tanh(ready_wait_time / (|slack|+eps)) activated only when slack > τ_safe → zero fairness pressure during violation risk
    - Uncertainty penalty applied multiplicatively to comm time only when min_incremental_energy < median_energy → targets edge VM risk
    - All weights derived from slack: urgency_weight = 1.0 + 3.0*sigmoid(-slack/τ), cfer_weight = sigmoid(slack/τ), etc.
    - Normalization uses median-absolute-deviation (MAD) with epsilon fallback for flat arrays; no percentile calls.
    """
    eps = 1e-8
    # Ensure float64 & safe casting
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    
    # Robust MAD-based normalization (degenerate-safe)
    def normalize_mad(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size == 1:
            return np.array([0.0])
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        scale = mad if mad > eps else (np.max(x) - np.min(x) + eps)
        scale = max(scale, eps)
        return (x - med) / scale
    
    τ = 2.0  # slack scale for soft gating
    τ_safe = 5.0  # slack margin threshold for fairness activation
    
    # Urgency: bounded, monotonic, high weight when slack negative
    sigmoid_urgency = 1.0 / (1.0 + np.exp(-slack / τ))  # [0,1], ↑ as slack ↓
    urgency_raw = -slack  # raw urgency: larger positive when more negative slack
    norm_urgency = normalize_mad(urgency_raw)
    urgency_weight = 1.0 + 3.0 * (1.0 - sigmoid_urgency)  # [1.0, 4.0]
    urgency_term = urgency_weight * norm_urgency
    
    # Criticality-Energy Ratio with Feasibility Gating (CF-CER)
    cfer_base = (upward_rank * remaining_work) / (min_incremental_energy + eps)
    feasibility_gate = 1.0 / (1.0 + np.exp(-slack / τ))  # smooth [0,1] gate: 0 when slack << 0, ~1 when slack >> 0
    cfer_masked = cfer_base * feasibility_gate
    norm_cfer = normalize_mad(cfer_masked)
    cfer_weight = 1.0 + 2.0 * sigmoid_urgency  # ↑ weight when urgency ↑ → prioritizes energy *only* when feasible
    cfer_term = -cfer_weight * norm_cfer  # negative: higher ratio → higher priority
    
    # Latency: split into execution-dominant (under urgency) and comm-dominant (under uncertainty + low energy)
    exec_latency = min_exec_time
    comm_latency = min_comm_time
    
    # Edge-risk amplification: apply uncertainty only to comm when energy is low (proxy for edge VM assignment)
    median_energy = np.median(min_incremental_energy) + eps
    edge_risk_mask = (min_incremental_energy < median_energy).astype(float)
    comm_with_risk = comm_latency * (1.0 + uncertainty * edge_risk_mask * 0.5)
    
    # Latency term = exec under urgency, comm+risk under safety
    latency_raw = np.where(slack < 0.0, exec_latency, comm_with_risk)
    norm_latency = normalize_mad(latency_raw)
    latency_weight = 0.7 + 0.8 * (1.0 - sigmoid_urgency)  # ↓ weight under urgency (deadline trumps latency)
    latency_term = latency_weight * norm_latency
    
    # Fairness: tanh-based aging, active only when slack > τ_safe
    fairness_raw = np.tanh(ready_wait_time / (np.abs(slack) + τ_safe + eps))
    fairness_active = (slack > τ_safe).astype(float)
    fairness_scaled = fairness_raw * fairness_active
    norm_fairness = normalize_mad(fairness_scaled)
    fairness_weight = 0.3 * sigmoid_urgency  # only mild fairness pressure when slack is safe
    fairness_term = -fairness_weight * norm_fairness  # negative: longer wait → higher priority
    
    # Assemble final score — smaller is better
    score = urgency_term + cfer_term + latency_term + fairness_term
    
    # Final safeguard: finite values only
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    
    return score
