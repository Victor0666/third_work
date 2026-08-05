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
    Priority rule v2 (evolved): Hybrid lexicographic DDL-hardened scoring with:
      - Hard urgency gating: strict dominance for slack < 0, no smoothing → ensures zero-tolerance DDL violation response
      - Criticality-Aware Energy Ratio (CAER): upward_rank * remaining_work / (min_incremental_energy + eps), masked only when slack >= 0 AND criticality > eps → prevents edge offloading of non-critical tasks while preserving energy efficiency for critical ones
      - Robust latency-aware fairness: ready_wait_time / (|slack| + eps) normalized via Q1/Q3 to prevent starvation under tight deadlines; always active (no slack>τ gate) → improves deadline density resilience
      - Uncertainty-weighted communication penalty: min_comm_time * (1 + uncertainty) used *only* when slack >= 0, otherwise min_exec_time dominates → reflects that late tasks prioritize compute over comms
      - Unified quantile normalization for all components → stable ranking under skew/outliers
      - Strict dominance hierarchy: urgency >> CAER >> latency >> fairness, enforced multiplicatively with decaying weights
      - All ops guarded against NaN/inf/zero at input, intermediate, and output levels
      - Final score clamped and sanitized for deterministic finite output
    """
    eps = 1e-8
    # Defensive input sanitization
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=1e6, neginf=-1e6)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=0.0, posinf=1e6, neginf=0.0)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=0.0, posinf=1e6, neginf=0.0)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=1e6, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=0.0, posinf=1e2, neginf=0.0)

    # Quantile-based robust normalization helper
    def robust_quantile_normalize(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size <= 1:
            return np.zeros_like(x)
        q1 = np.quantile(x, 0.25)
        q3 = np.quantile(x, 0.75)
        iqr = q3 - q1
        scale = iqr if iqr > eps else (np.max(x) - np.min(x) + eps)
        scale = max(scale, eps)
        center = (q1 + q3) / 2.0
        return (x - center) / scale

    # === 1. HARD URGENCY TERM (DDL VIOLATION DOMINANCE) ===
    # Binary urgency: 1 for violated, 0 otherwise → ensures strict priority for any slack < 0
    urgency_mask = (slack < 0).astype(float)
    # Normalize only non-zero cases to avoid division by zero in downstream logic
    urgency_norm = robust_quantile_normalize(urgency_mask)
    # Apply large multiplicative weight to enforce lexicographic dominance
    urgency_score = urgency_mask * 1e6 + (1.0 - urgency_mask) * urgency_norm * 1e3

    # === 2. CRITICALITY-AWARE ENERGY RATIO (CAER) ===
    # Only activate CAER when slack >= 0 (no violation) AND criticality is meaningful
    criticality = upward_rank * remaining_work
    caer_gate = (slack >= 0) & (criticality > eps)
    caer_base = criticality / (min_incremental_energy + eps)
    caer_masked = np.where(caer_gate, caer_base, 0.0)
    caer_norm = robust_quantile_normalize(caer_masked)
    # Invert to make higher CAER → lower score (prefer high-criticality + low-energy)
    caer_score = -caer_norm * 1e3

    # === 3. LATENCY TERM (CONTEXT-AWARE) ===
    # Under violation: prioritize execution time (fastest compute to catch up)
    # Under safety: penalize communication + uncertainty (edge offload risk)
    latency_raw = np.where(
        slack < 0,
        min_exec_time,
        min_comm_time * (1.0 + np.clip(uncertainty, 0.0, 10.0))
    )
    latency_norm = robust_quantile_normalize(latency_raw)
    latency_score = latency_norm * 1e2

    # === 4. UNCERTAINTY-AWARE FAIRNESS (STARVATION PREVENTION) ===
    # Always active: fairness pressure = wait_time / (|slack| + eps) → higher when slack shrinks
    fairness_raw = ready_wait_time / (np.abs(slack) + eps)
    fairness_norm = robust_quantile_normalize(fairness_raw)
    fairness_score = fairness_norm * 1e1

    # === COMBINE WITH STRICT HIERARCHY VIA ADDITIVE WEIGHTING ===
    # Urgency dominates (1e6), then CAER (1e3), latency (1e2), fairness (1e1)
    score = urgency_score + caer_score + latency_score + fairness_score

    # Final sanitization: ensure finite, deterministic, bounded output
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e9, 1e9)

    # Ensure shape (N,) even for N=1
    return score.astype(float)
