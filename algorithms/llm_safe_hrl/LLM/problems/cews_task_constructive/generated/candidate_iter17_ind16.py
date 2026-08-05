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
    Priority rule v2 (hybrid-evolved): Lexicographic urgency-gated energy-efficiency with
      - Robust arctan-based urgency: monotonic, bounded, and zero-centered at slack=0
      - SEER (energy/latency ratio) gated multiplicatively by exp(-max(0,-slack)/tau) → energy optimization only when safe
      - Risk-augmented criticality: upward_rank * remaining_work * (1 + tanh(uncertainty)) → boosts uncertain critical tasks smoothly
      - Tanh-based aging: saturates under high urgency, avoids starvation near deadline
      - MAD-normalized components with degenerate-safe fallback (no division by zero)
      - Final score = urgency_term * (1 + 0.4 * SEER_term) + risk_criticality_term + aging_term
      - All operations guarded against NaN/inf/zero; deterministic and finite output
    """
    eps = 1e-08
    # Safe conversion and nan/inf handling
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=1e6, neginf=-1e6)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=1e6, neginf=eps)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=0.0, posinf=10.0, neginf=0.0)

    # MAD normalization helper with degenerate-safe fallback
    def normalize_mad(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size == 1:
            return np.array([0.0])
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        if mad < eps:
            return np.zeros_like(x, dtype=float)
        return (x - med) / (mad + eps)

    # Urgency: arctan(-slack) → negative slack (lateness risk) yields large positive urgency_raw
    tau = 1.0
    urgency_raw = np.arctan(-slack / tau)  # monotonic, bounded in (-pi/2, pi/2)
    norm_urgency = normalize_mad(urgency_raw)
    # Scale to [0.1, 5.0] for strong dominance
    urgency_term = np.clip(1.0 + 4.0 * norm_urgency, 0.1, 5.0)

    # SEER (energy efficiency ratio): marginal energy per latency unit, gated only when slack < 0
    seer_base = (min_incremental_energy + eps) / (min_exec_time + min_comm_time + eps)
    slack_gate = np.exp(-np.maximum(0.0, -slack) / tau)  # 1.0 when slack >= 0, decays exponentially when slack < 0
    seer_gated = seer_base * slack_gate
    norm_seer = normalize_mad(seer_gated)
    # Invert so lower SEER (better efficiency) gives lower score → higher priority
    seer_term = np.clip(1.0 - 0.7 * norm_seer, 0.0, 1.0)

    # Risk-augmented criticality: preserve critical path importance under uncertainty
    risk_boost = 1.0 + np.tanh(uncertainty)  # bounded [1.0, 2.0]
    risk_criticality_base = upward_rank * remaining_work * risk_boost
    norm_criticality = normalize_mad(risk_criticality_base)
    # Negative weight: higher criticality → lower score (higher priority)
    risk_criticality_term = -0.8 * norm_criticality

    # Tanh-based aging: smooth saturation, suppressed when slack is highly negative
    wait_scale = 1.0 + np.abs(slack) + eps
    tanh_aging = np.tanh((ready_wait_time + eps) / wait_scale)
    norm_aging = normalize_mad(tanh_aging)
    aging_term = 0.12 * norm_aging

    # Lexicographic composition: urgency dominates multiplicatively; others additive
    score = urgency_term * (1.0 + 0.4 * seer_term) + risk_criticality_term + aging_term

    # Final sanitization: ensure finite, deterministic, shape-(N,)
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e9, 1e9)
    return score.astype(float)
