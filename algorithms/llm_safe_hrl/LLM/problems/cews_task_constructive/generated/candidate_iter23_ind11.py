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
    # Sanitize inputs: convert to float, replace NaN/inf with safe defaults
    eps = 1e-08
    def sanitize(x):
        x = np.asarray(x, dtype=float)
        x = np.nan_to_num(x, nan=eps, posinf=1e6, neginf=-1e6)
        return x
    min_exec_time = sanitize(min_exec_time)
    min_comm_time = sanitize(min_comm_time)
    min_incremental_energy = sanitize(min_incremental_energy)
    slack = sanitize(slack)
    upward_rank = sanitize(upward_rank)
    remaining_work = sanitize(remaining_work)
    ready_wait_time = sanitize(ready_wait_time)
    uncertainty = sanitize(uncertainty)

    # Robust normalization using IQR (stable, avoids outliers) — applied on full arrays only
    def robust_normalize(x):
        if x.size == 0:
            return np.zeros_like(x)
        if x.size == 1:
            return np.zeros_like(x)
        q25, q75 = np.percentile(x, [25, 75])
        iqr = q75 - q25 + eps
        med = np.median(x)
        normed = (x - med) / iqr
        return np.clip(normed, -4.0, 4.0)

    # === Deadline Violation Priority (Hard Constraint Enforcement) ===
    # Tasks with slack < -uncertainty are critically overdue → max priority (min score = 0.0)
    violation_mask = (slack < -uncertainty)
    base_score = np.full_like(slack, 1e6, dtype=float)

    # === Urgency Term: Tanh-scaled negative slack, normalized and clipped for stability ===
    # Use median absolute slack for adaptive scaling; avoid division by zero
    abs_slack = np.abs(slack)
    tau_urg = np.median(abs_slack) + eps
    tau_urg = np.maximum(tau_urg, 0.1)  # prevent too-small scale
    urgency_raw = np.tanh(-slack / tau_urg)  # [-1,1]; high when slack is very negative
    urgency_norm = robust_normalize(urgency_raw)
    # Map normalized urgency to [0.05, 2.0] — stronger boost for critical tasks
    urgency_term = 0.05 + 1.95 * (urgency_raw + 1.0) / 2.0

    # === Critical Path Density (CPD): upward_rank × remaining_work / latency, gated by rank percentile ===
    total_latency = min_exec_time + min_comm_time + eps
    # High-rank threshold: top 30% of upward_rank (more selective than median)
    uprank_thresh = np.percentile(upward_rank, 70) if upward_rank.size > 1 else np.mean(upward_rank)
    cpd_mask = (upward_rank >= uprank_thresh).astype(float)
    cpd_base = (upward_rank * remaining_work + eps) / total_latency
    cpd_gated = cpd_base * cpd_mask
    cpd_norm = robust_normalize(cpd_gated)
    # Smoothly map normalized CPD to [0.2, 1.8]
    cpd_term = 0.2 + 1.6 * np.clip((cpd_norm + 4.0) / 8.0, 0.0, 1.0)

    # === SEER (Scheduling Energy Efficiency Ratio): latency/energy, with deadline-aware gating ===
    seer_base = total_latency / (min_incremental_energy + eps)
    # Only activate energy-efficiency preference for non-overdue tasks (slack >= 0), reduced weight if uncertain
    seer_gate = np.where(slack >= 0, 1.0, 0.2)  # harsh penalty for overdue
    unc_damp = np.clip(1.0 - 0.4 * (uncertainty / (np.mean(uncertainty + eps) + eps)), 0.3, 1.0)
    seer_gated = seer_base * seer_gate * unc_damp
    seer_norm = robust_normalize(seer_gated)
    # Favor higher SEER → lower score; invert and compress
    seer_term = 0.3 + 1.4 * np.clip((3.0 - seer_norm) / 6.0, 0.0, 1.0)

    # === Fairness Boost: latency-aware wait fairness only for overdue or near-deadline tasks ===
    # Direct, low-overhead: reward long-waiting tasks *only* when slack is tight (<= 2s)
    tight_deadline_mask = (slack <= 2.0) & (slack > -1e-3)
    # Normalize wait relative to expected task duration to avoid bias toward slow tasks
    wait_ratio = ready_wait_time / (total_latency + eps)
    fairness_boost = np.where(
        tight_deadline_mask,
        np.clip(0.1 * wait_ratio * np.sqrt(np.maximum(2.0 - slack + eps, 0.0)), 0.0, 0.1),
        0.0
    )

    # === Composite score: multiplicative urgency × CPD × SEER, plus additive fairness boost ===
    composite = urgency_term * cpd_term * seer_term + fairness_boost

    # Apply violation mask: overdue-by-uncertainty tasks get highest priority (lowest score = 0.0)
    score = np.where(violation_mask, 0.0, composite)

    # Final sanitization: ensure finite, bounded, shape-(N,) output
    score = np.nan_to_num(score, nan=1e6, posinf=1e6, neginf=1e6)
    score = np.clip(score, 1e-8, 1e6)
    # Ensure shape is (N,) — no reshape needed if all ops preserve 1D
    return score
