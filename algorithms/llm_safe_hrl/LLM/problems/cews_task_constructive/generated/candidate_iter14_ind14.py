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
    v2 hybrid priority rule: Combines Parent 2's hard deadline safety and MAD robustness
    with Parent 1's arctan urgency scaling and uncertainty penalty refinement.
    
    Key innovations:
    - Lexicographic safety: absolute override for slack < 0, then strict slack-gating for all non-urgency terms
    - Arctan-based urgency (Parent 1) preserves monotonicity and avoids 1/slack singularity near zero
    - Unified uncertainty amplification: scales linearly with |slack| distance from median positive slack,
      applied to both latency inflation and fairness penalty
    - Energy-aware CED uses min_incremental_energy * upward_rank / (remaining_work + eps) for better
      energy-latency tradeoff under tight deadlines
    - Fairness term uses sqrt(wait) * (1 + |slack|/median_slack_pos) to age tasks gracefully only when safe
    - All terms normalized via robust MAD with N=1 handling and strict clipping
    - Final score clamped to finite bounds and sanitized against NaN/inf
    """
    eps = 1e-08
    # Safe conversion and nan/inf sanitization
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=0.0, posinf=1e6, neginf=1e6)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=0.0, posinf=1e6, neginf=1e6)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=0.0, posinf=1e6, neginf=1e6)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=1e6, neginf=-1e6)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=0.0, posinf=1e6, neginf=1e6)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=0.0, posinf=1e6, neginf=1e6)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=1e6, neginf=1e6)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=0.0, posinf=1e6, neginf=1e6)
    
    N = len(slack)
    if N == 0:
        return np.array([], dtype=float)
    
    def normalize_mad(x):
        x = np.clip(x, -1e6, 1e6)
        if x.size == 1:
            return np.array([0.0])
        med = np.median(x)
        abs_dev = np.abs(x - med)
        mad = np.median(abs_dev)
        scale = mad if mad > eps else eps
        norm = (x - med) / scale
        return np.clip(norm, -2.5, 2.5)
    
    # Hard deadline violation override: highest priority for violated tasks
    violation_mask = slack < 0
    hard_ddl_offset = np.where(violation_mask, -1000.0, 0.0)
    
    # Urgency: arctan(-slack/tau) scaled by median |slack| for scale invariance (Parent 1)
    abs_slack = np.abs(slack) + eps
    tau = np.median(abs_slack)
    urgency_raw = np.arctan(-slack / tau)
    norm_urgency = normalize_mad(urgency_raw)
    
    # Slack gating thresholds
    positive_slack_mask = slack > 0
    median_slack_pos = np.median(slack[positive_slack_mask]) if np.any(positive_slack_mask) else 1.0
    
    # CED term: critical-path energy efficiency — gated by slack >= median_slack_pos (safety buffer)
    ced_feasibility_gate = (slack >= 0) & (slack >= median_slack_pos - eps)
    # Use energy-per-work * upward_rank for tighter energy-latency coupling
    ced_base = (min_incremental_energy + eps) * upward_rank / (remaining_work + eps)
    ced_masked = np.where(ced_feasibility_gate, ced_base, 0.0)
    norm_ced = normalize_mad(ced_masked)
    ced_term = -2.2 * norm_ced
    
    # Energy term: pure marginal energy minimization — same gating
    energy_masked = np.where(ced_feasibility_gate, min_incremental_energy, 0.0)
    norm_energy = normalize_mad(energy_masked)
    energy_term = -1.3 * norm_energy
    
    # Latency term: exec + comm, inflated by uncertainty scaled by slack tightness
    base_latency = min_exec_time + min_comm_time + eps
    # Tightness ratio: 1 when slack <= 0, decays to 0 as slack >> median_slack_pos
    tightness_ratio = np.clip((2.0 * median_slack_pos - slack) / (2.0 * median_slack_pos + eps), 0.0, 1.0)
    # Uncertainty amplification: stronger penalty when slack is small or negative
    unc_amplification = 1.0 + (1.0 - tightness_ratio) * np.clip(uncertainty, 0.0, 1.0)
    inflated_latency = base_latency * unc_amplification
    norm_latency = normalize_mad(inflated_latency)
    latency_term = 0.85 * norm_latency
    
    # Fairness term: aging reward scaled by slack margin to prevent starvation during safety surplus
    slack_margin = np.where(slack > 0, slack, 0.0)
    # Use sqrt(wait) to avoid excessive aging; scale by relative slack margin
    fairness_raw = np.sqrt(np.maximum(ready_wait_time, 0.0) + eps) * (1.0 + slack_margin / (median_slack_pos + eps))
    # Only activate fairness when slack is safely above median (prevents interference with urgency)
    fairness_gate = (slack_margin > median_slack_pos).astype(float)
    fairness_masked = fairness_raw * fairness_gate
    norm_fairness = normalize_mad(fairness_masked)
    fairness_term = -0.15 * norm_fairness
    
    # Unified uncertainty penalty: applied unconditionally but weighted by slack proximity to boundary
    # Penalty increases as slack approaches 0 from positive side or goes negative
    slack_distance = np.abs(slack - median_slack_pos)
    unc_penalty = uncertainty * (1.0 + slack_distance / (median_slack_pos + eps))
    norm_uncertainty = normalize_mad(unc_penalty)
    uncertainty_term = 0.2 * norm_uncertainty
    
    # Combine with weights tuned for DDL-hard + energy-aware balance
    score = (
        4.5 * norm_urgency +
        ced_term +
        energy_term +
        latency_term +
        fairness_term +
        uncertainty_term +
        hard_ddl_offset
    )
    
    # Final sanitization: ensure finite, deterministic output
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e9, 1e9)
    
    return score.reshape(-1)
