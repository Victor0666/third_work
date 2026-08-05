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
    Improved priority rule combining robustness of v0 and stability/interpretability of v1.
    Key improvements:
    - Uses arctan-based deadline risk (bounded, smooth) from v0 but only for negative slack (v1's cleaner gating).
    - Critical-energy efficiency is sigmoid-bounded (v1) AND gated by slack > 0 (both), avoiding noise.
    - Introduces *latency-criticality ratio*: (exec+comm) / upward_rank — penalizes high-latency low-importance tasks.
    - Wait fairness uses clipped inverse-slack scaling (not percentile) for monotonic urgency-aware boosting.
    - Uncertainty inflates latency *and* modulates deadline risk sensitivity via slack-dependent gain.
    - All terms normalized via robust MAD with explicit zero-degenerate handling; final score strictly finite.
    - Strict DDL-first hierarchy enforced: deadline term dominates (largest weight), others only refine under feasibility.
    """
    eps = 1e-08
    # Convert inputs to float arrays without modifying in-place
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    def safe_mad_normalize(x):
        """Robust MAD normalization: handles N=1, constant, NaN, inf. Returns zeros if degenerate."""
        if x.size == 0:
            return np.zeros_like(x)
        x_clean = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x_clean.size == 1:
            return np.zeros_like(x_clean)
        med = np.median(x_clean)
        abs_dev = np.abs(x_clean - med)
        mad = np.median(abs_dev) + eps
        normed = (x_clean - med) / mad
        return np.clip(normed, -4.0, 4.0)

    # === 1. Deadline Risk (DDL-first anchor) ===
    # Only activate arctan penalty for overdue/near-deadline tasks; bounded [-π/2, 0] for slack <= 0
    deadline_risk_raw = np.where(slack <= 0, np.arctan(-slack), 0.0)
    deadline_score = safe_mad_normalize(deadline_risk_raw)

    # === 2. Critical-Energy Efficiency (energy-aware refinement under slack > 0) ===
    # Bounded sigmoid on (upward_rank * remaining_work / energy) — higher ratio = better efficiency
    energy_efficiency_raw = np.where(
        slack > 0,
        (upward_rank * remaining_work + eps) / (min_incremental_energy + eps),
        0.0
    )
    # Sigmoid compression: maps R → (0,1), centered & scaled for stability
    energy_efficiency_sig = 1.0 / (1.0 + np.exp(-np.clip(energy_efficiency_raw, -6.0, 6.0) * 0.25))
    energy_efficiency_norm = safe_mad_normalize(energy_efficiency_sig)

    # === 3. Latency-Criticality Ratio (penalizes high-latency, low-importance work) ===
    # Encourages scheduling of critical tasks *before* they become bottlenecks; avoids idle high-energy VMs
    total_latency = np.clip(min_exec_time + min_comm_time, eps, None)
    latency_criticality_ratio = np.where(
        upward_rank > eps,
        total_latency / (upward_rank + eps),
        total_latency / eps  # fallback: treat zero-rank as very non-critical
    )
    latency_crit_norm = safe_mad_normalize(latency_criticality_ratio)

    # === 4. Urgency-Aware Wait Boost (fairness without undermining deadlines) ===
    # Boosts long-waiting tasks *only when slack margin exists*, scaled inversely to slack (more boost when slack is ample)
    slack_margin = np.clip(slack, 0.0, None) + eps
    wait_boost_base = np.sqrt(np.clip(ready_wait_time, 0.0, None))
    # Clipped ratio: [0, 0.25], increases as slack_margin shrinks → more boost when margin is tight but still positive
    wait_boost = np.clip(wait_boost_base / (slack_margin + eps), 0.0, 0.25)

    # === 5. Uncertainty-Aware Execution Risk ===
    # Inflates latency *and* amplifies deadline risk sensitivity when slack is low
    uncertainty_factor = np.clip(uncertainty, 0.0, 3.0)
    inflated_latency = total_latency * (1.0 + uncertainty_factor * 0.15)
    inflated_lat_norm = safe_mad_normalize(inflated_latency)
    
    # Modulated deadline gain: increase deadline_score weight when uncertainty is high *and* slack is low
    deadline_gain_mod = 1.0 + (uncertainty_factor * np.clip(-slack, 0.0, None) * 0.05)

    # === Composition: hierarchical weighting with DDL dominance ===
    # Deadline term carries highest weight and is modulated by uncertainty; others are refinements
    score = (
        +5.0 * deadline_score * deadline_gain_mod  # Primary: hard DDL enforcement
        - 2.5 * energy_efficiency_norm              # Secondary: maximize critical work per joule
        + 1.2 * latency_crit_norm                   # Tertiary: avoid low-value high-latency tasks
        + 0.18 * wait_boost                         # Fairness: mild starvation mitigation
        + 0.3 * inflated_lat_norm                   # Risk: account for uncertainty-inflated delays
    )

    # Final sanitization: ensure finite, deterministic output
    score = np.nan_to_num(
        score,
        nan=1e12,
        posinf=1e12,
        neginf=-1e12
    )
    
    # Enforce shape (N,) — no squeezing or scalar return
    return score.reshape(-1)
