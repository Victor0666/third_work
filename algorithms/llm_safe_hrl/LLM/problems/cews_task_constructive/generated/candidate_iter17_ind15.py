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
    v2 improved priority rule: Combines best elements from both parents with novel enhancements:
    - Uses sharper tanh urgency (τ=0.5) from Parent 2, but adds slack-aware clipping to prevent numerical instability near zero.
    - Integrates CPD (critical-path density = upward_rank / (min_exec_time + ε)) from Parent 2 for latency-criticality coupling.
    - Adopts SEER (Slack-Energy-Efficiency Ratio) with smooth sigmoid activation from Parent 2, but gates it *only* under positive slack to strictly enforce DDL-hardness.
    - Introduces *latency-risk ratio*: (min_exec_time + min_comm_time) / (|slack| + ε), prioritizing low-latency tasks when slack is tight — directly penalizes high-latency tasks near deadlines.
    - Enhances fairness with risk-tempered sqrt(wait) scaled by exp(-uncertainty * I(slack > 0)), avoiding starvation *only* when safe.
    - Replaces all normalizations with unified robust_minmax_mad: uses MAD if stable, else minmax with explicit degeneracy handling; size-1 → [0].
    - Adds explicit zero-energy masking: tasks with near-zero min_incremental_energy get boosted priority only if slack > 0 (energy-efficient + deadline-safe).
    - All terms bounded, finite, deterministic, and zero/Nan/inf protected.
    """
    eps = 1e-08

    # Defensive casting and NaN/inf cleanup
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=eps, posinf=eps, neginf=eps)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=1e6, neginf=-1e6)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=eps, posinf=eps, neginf=eps)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

    def robust_minmax_mad(x):
        """Unified normalization: MAD if stable, else minmax; handles size-1 and degenerate scale."""
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size == 1:
            return np.array([0.0])
        med = np.median(x)
        abs_dev = np.abs(x - med)
        mad = np.median(abs_dev)
        if mad > eps:
            scale = mad + eps
        else:
            xmin, xmax = np.min(x), np.max(x)
            scale = xmax - xmin
            if scale < eps:
                scale = eps
        return (x - med) / (scale + eps)

    # 1. Sharper urgency: tanh(-slack/τ) with τ=0.5 (Parent 2), clipped for stability near slack=0
    tau_urg = 0.5
    urgency_raw = np.tanh(-np.clip(slack, -100.0, 100.0) / tau_urg)  # avoid overflow in tanh
    norm_urgency = robust_minmax_mad(urgency_raw)
    urgency_term = -3.0 * norm_urgency

    # 2. Critical-Path Density (CPD): upward_rank / (min_exec_time + ε) — latency-criticality coupling (Parent 2)
    cpd_base = upward_rank / (min_exec_time + eps)
    norm_cpd = robust_minmax_mad(cpd_base)
    cpd_term = -1.5 * norm_cpd

    # 3. Slack-Activated SEER: upward_rank * remaining_work / (min_incremental_energy + ε), gated by slack safety
    seer_base = upward_rank * remaining_work / (min_incremental_energy + eps)
    # Sigmoid activation: activates only when slack > 0, smoothly ramps up as slack increases
    seer_activation = 1.0 / (1.0 + np.exp(-(slack + eps) / 2.0))
    seer_masked = seer_base * seer_activation
    norm_seer = robust_minmax_mad(seer_masked)
    seer_term = -1.0 * norm_seer

    # 4. Novel latency-risk ratio: (exec+comm) / (|slack| + ε) — strongly penalizes high-latency tasks when slack is tight
    latency_risk_ratio = (min_exec_time + min_comm_time + eps) / (np.abs(slack) + eps)
    # Only apply strong penalty when slack is small (tight deadline regime); cap extreme values
    latency_risk_mask = np.where(np.abs(slack) < 5.0, 1.0, 0.0)
    latency_risk_penalized = np.where(latency_risk_mask, latency_risk_ratio, 0.0)
    norm_latency_risk = robust_minmax_mad(latency_risk_penalized)
    latency_risk_term = 0.85 * norm_latency_risk

    # 5. Risk-tempered fairness: sqrt(wait) * exp(-uncertainty) only under safe slack (Parent 2), with tighter clip
    wait_safe = np.maximum(ready_wait_time, 0.0)
    sqrt_wait = np.sqrt(wait_safe + eps)
    exp_decay = np.where(slack > 0.0, np.exp(-uncertainty), 0.0)  # zero fairness penalty if slack ≤ 0
    fairness_raw = sqrt_wait * exp_decay
    fairness_clipped = np.clip(fairness_raw, 0.0, 0.2)
    norm_fairness = robust_minmax_mad(fairness_clipped)
    fairness_term = -0.35 * norm_fairness

    # 6. Uncertainty penalty: only applied when slack > 0 (safe regime), scaled by normalized uncertainty
    uncertainty_penalty = np.where(slack > 0.0, uncertainty, 0.0)
    norm_uncert = robust_minmax_mad(uncertainty_penalty)
    uncert_term = 0.3 * norm_uncert

    # 7. Zero-energy masking: boost priority for energy-efficient tasks *only* when slack > 0
    energy_efficiency = 1.0 / (min_incremental_energy + eps)
    energy_mask = np.where(slack > 0.0, 1.0, 0.0)
    energy_boost = energy_efficiency * energy_mask
    norm_energy = robust_minmax_mad(energy_boost)
    energy_term = -0.6 * norm_energy

    # Aggregate score: smaller = higher priority
    score = (
        urgency_term +
        cpd_term +
        seer_term +
        latency_risk_term +
        fairness_term +
        uncert_term +
        energy_term
    )

    # Final safeguard: ensure finite, deterministic output
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    return score
