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
    Priority rule v2: Hybrid of Parent 1's criticality-aware latency coupling and Parent 2's robust risk-amplified energy/criticality,
    with hard-deadline enforcement, uncertainty-gated starvation mitigation, and monotonic quantile normalization for rank preservation.
    
    Key innovations:
      - Slack penalty: Hard exponential for violations + asymmetric sigmoid for tight slack (Parent 2), but scaled by remaining_work to reflect workload severity
      - Critical energy term: `min_incremental_energy / (upward_rank + eps) * (1 + uncertainty)` (Parent 2), now weighted by slack urgency mask to avoid penalizing early non-critical tasks
      - Latency coupling: `min_comm_time * (1 + 0.5 * upward_rank / (np.median(upward_rank) + eps))` (Parent 1 inspiration), normalized separately
      - Starvation index: `ready_wait_time / (np.clip(slack, 0, 60) + eps)` — only activated under deadline pressure (Parent 2), but uses ratio instead of clipping for smoother gradient
      - Uncertainty gating: multiplicative on both slack penalty and critical energy (Parent 2), but attenuated when slack > 120s (low-risk regime)
      - Normalization: quantile-based rank mapping (Parent 1) for strict ordering preservation, with degenerate fallback for N=1
      - Final dominance: urgency dominates (0.55), then risk-weighted critical energy (0.3), latency coupling (0.1), starvation (0.05)
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

    # Quantile-based rank-preserving normalization (Parent 1 style, robust for all N)
    def quantile_map(x):
        if len(x) == 1:
            return np.array([0.5])
        ranks = np.argsort(np.argsort(x)).astype(float)
        return 0.05 + 0.9 * (ranks / (len(x) - 1 + eps))
    
    # Slack penalty: hard exponential for violations, asymmetric sigmoid for tight slack, scaled by workload severity
    slack_penalty = np.zeros_like(slack)
    violated_mask = slack < 0
    tight_mask = ~violated_mask & (slack < 60.0)
    # Exponential urgency for violations: grows unbounded as slack becomes more negative
    slack_penalty[violated_mask] = np.exp(-slack[violated_mask])
    # Sigmoid decay for tight slack: steeper near deadline (30s), shallower beyond
    slack_penalty[tight_mask] = 1.0 / (1.0 + np.exp((slack[tight_mask] - 30.0) / 5.0))
    # Scale by remaining work to prioritize high-impact violations
    work_scale = np.clip(remaining_work / (np.median(remaining_work) + eps), 0.1, 10.0)
    slack_penalty = slack_penalty * work_scale
    
    # Critical energy: marginal energy per unit criticality, amplified by uncertainty
    critical_energy_base = min_incremental_energy / (upward_rank + eps)
    critical_energy_score = critical_energy_base * (1.0 + uncertainty)
    
    # Latency coupling: communication time weighted by relative criticality (upward_rank normalized)
    median_ur = np.median(upward_rank) + eps
    comm_latency_weighted = min_comm_time * (1.0 + 0.5 * upward_rank / median_ur)
    
    # Starvation index: wait time normalized by slack margin (only meaningful under pressure)
    starvation_raw = np.where(slack <= 60.0, ready_wait_time / (np.clip(slack, 0.0, 60.0) + eps), 0.0)
    
    # Uncertainty gating: multiplicative risk amplification only in risky regimes (slack < 120s)
    risk_mask = slack < 120.0
    risk_factor = np.where(risk_mask, 1.0 + 0.5 * uncertainty, 1.0)
    
    # Normalize each component independently with rank mapping
    slack_norm = quantile_map(slack_penalty)
    energy_norm = quantile_map(critical_energy_score)
    comm_norm = quantile_map(comm_latency_weighted + eps)
    starv_norm = quantile_map(starvation_raw)
    
    # Apply risk factor to dominant terms only
    slack_score = slack_norm * risk_factor
    energy_score = energy_norm * risk_factor
    
    # Final score: strict dominance hierarchy with calibrated weights
    # Urgency (slack) is primary driver; critical energy secondary; latency and fairness tertiary
    score = (
        0.55 * slack_score +
        0.30 * energy_score +
        0.10 * comm_norm +
        0.05 * starv_norm
    )
    
    # Ensure finite output and clamp extremes deterministically
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e9, 1e9)
    
    return score
