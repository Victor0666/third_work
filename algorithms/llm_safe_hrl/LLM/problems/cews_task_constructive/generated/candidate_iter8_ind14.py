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
    v2 crossover: Combines Parent 2's monotonic urgency & slack-aware aging with Parent 1's
    critical-energy density gating and robust min-max normalization. Introduces:
      - Unified deadline-energy tradeoff via slack-gated CED weight scaling
      - Uncertainty-inflated latency for efficiency term (not just additive penalty)
      - MAD-based robust normalization with explicit N=1/constant fallbacks
      - Bounded hard penalty capped at 0.5 to prevent dominance over energy optimization
      - Aging boost scaled by clipped sqrt(wait) *and* sigmoid(slack pressure) for fairness
    Ensures strict DDL adherence first, then optimizes energy per criticality under margin.
    """
    eps = 1e-08
    # Ensure float arrays, no in-place mutation
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    # === Deadline Urgency: smooth, monotonic, bounded [0,1] ===
    # arctan(-slack) maps (-∞,∞) → (0, π), normalized to [0,1]; handles slack=0 continuously
    deadline_urgency = (np.arctan(-slack) + np.pi / 2) / np.pi

    # === Hard Penalty: only for violated deadlines, bounded to avoid score explosion ===
    hard_penalty = np.where(slack < -eps, 
                            np.clip(-slack / (np.abs(np.min(slack)) + eps), 0.0, 0.5), 
                            0.0)

    # Combined deadline risk score before normalization
    deadline_risk = deadline_urgency + hard_penalty

    # === Robust MAD Normalization (handles N=1, constants, NaN/inf) ===
    def robust_mad_normalize(x):
        x_clean = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x_clean.size == 1:
            return np.zeros_like(x_clean)
        median_x = np.median(x_clean)
        mad = np.median(np.abs(x_clean - median_x)) + eps
        if mad < eps:
            return np.zeros_like(x_clean)
        normed = (x_clean - median_x) / mad
        return np.clip(normed, -3.0, 3.0)

    deadline_score = robust_mad_normalize(deadline_risk)

    # === Critical-Energy Density (CED): upward_rank * work / energy, but ONLY when slack >= 0 ===
    base_ced = upward_rank * remaining_work / (min_incremental_energy + eps)
    ced_masked = np.where(slack >= 0, base_ced, 0.0)
    ced_norm = robust_mad_normalize(ced_masked)

    # === Efficiency Term: energy per *uncertainty-inflated latency* under positive slack ===
    base_latency = min_exec_time + min_comm_time + eps
    # Inflate latency only for tasks with tight-but-positive slack (0 < slack <= median_positive)
    median_pos_slack = np.median(slack[slack > 0]) if np.any(slack > 0) else np.percentile(np.abs(slack), 75) + eps
    latency_inflation_active = (slack > 0) & (slack <= median_pos_slack + eps)
    inflated_latency = np.where(latency_inflation_active,
                               base_latency + np.clip(uncertainty, 0.0, 0.5 * np.percentile(base_latency, 90)),
                               base_latency)
    # Energy-per-inflated-latency: higher value = better efficiency (so we negate for priority)
    energy_per_latency = min_incremental_energy / (inflated_latency + eps)
    eff_gated = np.where(slack >= 0, energy_per_latency, 0.0)
    eff_norm = robust_mad_normalize(eff_gated)

    # === Aging Boost: sqrt(wait) * decay factor to prevent starvation under deadline pressure ===
    sqrt_wait = np.sqrt(np.maximum(ready_wait_time, 0.0) + eps)
    # Decay factor: 1 when slack is large, ~0 when slack is negative or zero
    slack_pressure = np.clip((0.0 - slack) / (np.abs(np.min(slack)) + eps), 0.0, 1.0)
    aging_decay = 1.0 / (1.0 + np.exp(4.0 * (slack_pressure - 0.5)))  # steep sigmoid around slack=0
    aging_boost = 0.03 * np.clip(sqrt_wait, 0.0, np.percentile(sqrt_wait, 90) + eps) * aging_decay

    # === Final score: smaller = higher priority ===
    # Deadline dominates (positive weight), CED & efficiency reduce score (negative weights), aging slightly helps long-waiting
    score = (
        +3.8 * deadline_score           # Strong urgency enforcement
        - 2.6 * ced_norm               # Prioritize high-impact/joule under margin
        - 0.9 * eff_norm               # Favor energy-efficient execution where safe
        + 0.03 * aging_boost           # Mild starvation prevention
    )

    # Final sanitization: ensure finite, deterministic output
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    
    # Ensure shape (N,) even for N=1
    return score.reshape(-1)
