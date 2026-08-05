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
    v2: Hybrid priority rule combining strict deadline enforcement (v1) with robust normalization (v0),
         enhanced by latency-aware critical-energy density, uncertainty-gated urgency, and starvation-robust fairness.
    
    Key improvements:
    - Uncertainty-modulated urgency: slack is penalized by uncertainty *only when slack < 0, scaled smoothly via tanh to avoid explosion.
    - Latency-aware CED: uses (min_exec_time + min_comm_time + eps) denominator, but multiplies by upward_rank * remaining_work only if slack > 0 and both are positive.
    - Unified safe_mad_normalize with degenerate-case fallbacks and explicit outlier clipping before MAD computation.
    - Fairness via log1p-scaled wait time, capped at 0.1 and weighted conservatively (0.05) to avoid overriding deadline signals.
    - Hard urgency override remains for severe violations (slack < -10), but softens transition using tanh-based risk amplification.
    - All terms strictly bounded; final score clipped to finite range to ensure determinism and stability.
    """
    eps = 1e-08
    
    # Clean inputs: replace NaN/inf with safe finite defaults
    def clean_array(x):
        x = np.asarray(x, dtype=float)
        return np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=eps)
    
    min_exec_time = clean_array(min_exec_time)
    min_comm_time = clean_array(min_comm_time)
    min_incremental_energy = clean_array(min_incremental_energy)
    slack = clean_array(slack)
    upward_rank = clean_array(upward_rank)
    remaining_work = clean_array(remaining_work)
    ready_wait_time = clean_array(ready_wait_time)
    uncertainty = clean_array(uncertainty)
    
    def safe_mad_normalize(x):
        """Robust MAD normalization handling N=1, constants, outliers"""
        if x.size == 1:
            return np.zeros_like(x)
        # Clip extremes to prevent MAD collapse
        x_clipped = np.clip(x, -1e6, 1e6)
        med = np.median(x_clipped)
        mad = np.median(np.abs(x_clipped - med)) + eps
        if mad < eps:
            return np.zeros_like(x)
        normed = (x_clipped - med) / mad
        return np.clip(normed, -4.0, 4.0)
    
    # --- Urgency: uncertainty-modulated, three-tier with smooth transitions ---
    # Base urgency signal: tanh-based scaling for negative slack, preserving monotonicity
    raw_urgency = np.where(slack < 0, np.tanh(-slack / 2.0), 0.0)
    # Amplify urgency using uncertainty only when slack < 0, smoothly bounded
    uncertainty_factor = np.clip(uncertainty, 0.0, 10.0)
    urgency_risk = np.where(slack < 0, raw_urgency * (1.0 + 0.3 * uncertainty_factor), 0.0)
    
    # Severe violation override: hard priority for slack < -10
    is_severe = slack < -10.0
    urgency_raw = np.where(is_severe, 1000.0, urgency_risk)
    
    # Normalize urgency — higher urgency → lower priority score (so invert sign later)
    urgency_norm = safe_mad_normalize(urgency_raw)
    
    # --- Critical-Energy Density (CED): latency-aware & deadline-gated ---
    exec_comm_lat = min_exec_time + min_comm_time + eps
    ced_numerator = upward_rank * remaining_work
    # Only activate CED when slack > 0 AND both rank/work are meaningful
    ced_active = (slack > 0) & (upward_rank > eps) & (remaining_work > eps)
    ced_raw = np.where(ced_active, ced_numerator / (min_incremental_energy * exec_comm_lat + eps), 0.0)
    ced_norm = safe_mad_normalize(ced_raw)
    
    # --- Upward rank contribution: gated by slack > 0 and non-zero work ---
    upward_active = np.where(ced_active, upward_rank, 0.0)
    upward_norm = safe_mad_normalize(upward_active)
    
    # --- Risk penalty: uncertainty × |slack| only when slack < 0, capped ---
    risk_raw = np.where(slack < 0, np.clip(-slack * uncertainty_factor, 0.0, 100.0), 0.0)
    risk_norm = safe_mad_normalize(risk_raw)
    
    # --- Fairness: log1p-scaled wait time, capped and softly applied ---
    max_wait = np.max(ready_wait_time) + eps
    rel_log_wait = np.log1p(ready_wait_time) / (np.log1p(max_wait) + eps)
    fairness_boost = np.clip(rel_log_wait, 0.0, 0.1)
    
    # --- Composite score: deadline dominates, then energy efficiency, then fairness ---
    # Sign convention: smaller score = higher priority → urgency should *lower* score (hence -urgency_norm)
    # CED and upward_rank reflect importance: higher values mean *more critical*, so we want them prioritized → subtract
    # Risk penalty reflects danger: higher risk should *increase* score (i.e., deprioritize) → add
    score = (
        -5.0 * urgency_norm           # Strongest weight: urgent tasks get top priority
        -2.4 * ced_norm              # Energy-latency efficiency: prefer low-energy-per-latency critical tasks
        -1.3 * upward_norm           # Structural importance: only when deadline-safe
        + 0.35 * risk_norm           # Penalty for violation severity + uncertainty
        + 0.05 * fairness_boost      # Mild boost for long-waiting tasks (prevents starvation)
    )
    
    # Final guard: ensure finite, deterministic output
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e9, 1e9)
    
    return score
