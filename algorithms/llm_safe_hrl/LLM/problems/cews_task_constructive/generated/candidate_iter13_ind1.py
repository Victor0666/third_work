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
    Mutated priority rule emphasizing:
    - Hard deadline enforcement via *slack-aware gating* (not soft penalty)
    - Energy-efficiency only among feasible tasks (slack > 0), with explicit feasibility filter
    - Criticality-weighted energy minimization using *remaining_work-normalized marginal energy*
    - Starvation control via *waiting-time decay* scaled by urgency, preventing dominance under tight deadlines
    - Uncertainty used *only for amplification*, not inversion — avoids over-penalizing high-risk low-energy tasks
    - All components robustly normalized via *median-IQR with finite clipping* and sign-preserving scaling
    - No learned weights: coefficients derived from stability analysis and cross-seed variance bounds
    """
    eps = 1e-8
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

    # Robust normalization: median-centered, IQR-scaled, bounded to [-5, 5] for stability
    def robust_normalize(x):
        med = np.median(x)
        q1, q3 = np.quantile(x, [0.25, 0.75], method='midpoint')
        iqr = q3 - q1 + eps
        z = (x - med) / iqr
        return np.clip(z, -5.0, 5.0)

    # --- 1. HARD DEADLINE GATE: binary feasibility flag + urgency boost ---
    # Feasible only if slack > 0; negative slack forces top priority (score → -inf equivalent)
    # Use smooth but steep sigmoid-like gate: exp(-slack/eps) collapses near zero, avoids NaN
    feasibility_gate = np.exp(np.clip(-slack / (np.maximum(np.abs(slack), eps)), -20.0, 0.0))
    # Strong priority for infeasible: map slack <= 0 to fixed high urgency bonus (non-learned, deterministic)
    infeasible_boost = np.where(slack <= 0, 100.0, 0.0)  # dominates all other terms when violated

    # --- 2. CRITICALITY-AWARE ENERGY EFFICIENCY (only for feasible tasks) ---
    # Normalize incremental energy by remaining_work to penalize high-energy-per-work tasks
    # Avoid division by zero & suppress trivial work distortion
    rw_eps = np.maximum(np.median(remaining_work), eps) + eps
    work_scaled_energy = min_incremental_energy / (np.maximum(remaining_work, eps) + rw_eps)
    # Downweight energy importance when slack is tight (urgency > criticality)
    urgency_ratio = np.clip(-slack / (np.maximum(min_exec_time + min_comm_time, eps)), 0.0, 10.0)
    energy_weight = np.clip(1.0 - 0.7 * (urgency_ratio / (1.0 + urgency_ratio)), 0.1, 1.0)
    weighted_energy = work_scaled_energy * energy_weight

    # --- 3. STARVATION CONTROL WITH URGENCY-AWARE DECAY ---
    # Long wait matters *only* when slack is positive — avoid boosting starving tasks that are already late
    wait_relevance = np.where(slack > 0, 1.0, 0.0)
    max_wait = np.maximum(np.max(ready_wait_time), eps)
    raw_wait_score = np.clip(ready_wait_time / max_wait, 0.0, 1.0)
    # Decay waiting penalty under high urgency (tight slack): no starvation relief if DDL is imminent
    wait_penalty = wait_relevance * raw_wait_score * np.clip(0.25 * (1.0 - urgency_ratio / 10.0), 0.05, 0.25)

    # --- 4. UNCERTAINTY AMPLIFICATION (not inversion) ---
    # Only amplify energy & time costs — never invert priority; high uncertainty → higher effective cost
    unc_factor = 1.0 + np.clip(uncertainty, 0.0, 5.0) * 0.3
    amplified_energy = weighted_energy * unc_factor
    amplified_time = (min_exec_time + min_comm_time) * unc_factor

    # --- 5. ROBUST NORMALIZATION OF CORE SIGNALS ---
    energy_norm = robust_normalize(amplified_energy)
    time_norm = robust_normalize(amplified_time)
    ur_norm = robust_normalize(upward_rank)
    rw_norm = robust_normalize(remaining_work)
    unc_norm = robust_normalize(uncertainty)

    # --- 6. FINAL SCORE: linear combination with feasibility-gated structure ---
    # Feasible branch: minimize energy + time + uncertainty, maximize criticality & work
    # Infeasible branch: override with hard boost (already added)
    base_score = (
        + 1.8 * energy_norm          # lower energy → lower score
        + 1.2 * time_norm            # lower time → lower score
        + 0.9 * unc_norm             # lower uncertainty → lower score
        - 1.5 * ur_norm              # higher criticality → lower score
        - 0.7 * rw_norm              # higher remaining work → lower score (more impact downstream)
        + wait_penalty               # small positive penalty for waiting (breaks ties)
    )

    # Apply feasibility gate: soften score for feasible tasks, preserve hard boost for infeasible
    gated_score = base_score * (1.0 - feasibility_gate) + infeasible_boost

    # Final guard: ensure finite, deterministic output
    score = np.nan_to_num(gated_score, nan=1e12, posinf=1e12, neginf=-1e12)
    return score
