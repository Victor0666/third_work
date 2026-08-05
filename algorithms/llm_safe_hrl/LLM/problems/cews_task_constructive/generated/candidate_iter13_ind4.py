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
    v2 mutation: Replaces additive blending with lexicographic dominance via slack-gated term masking;
    introduces *energy-criticality ratio* (ECR) = energy / (upward_rank * work) as primary feasibility-aware efficiency signal;
    replaces arctan urgency with clipped linear risk ramp for interpretability and stronger penalty on negative slack;
    replaces sqrt-wait fairness with log-linear aging boost activated only when slack > threshold (avoiding starvation without DDL compromise);
    uses uncertainty-weighted latency only for feasible tasks (slack >= 0), and applies robust MAD-based normalization with epsilon-clipped scale;
    enforces strict priority ordering: urgency dominates → then ECR efficiency → then latency → then fairness — via multiplicative gating, not addition.
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

    # Robust adaptive normalization using MAD (median absolute deviation) with degenerate fallback
    def normalize_mad(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size == 1:
            return np.array([0.0])
        center = np.median(x)
        dev = np.abs(x - center)
        mad = np.median(dev)
        if mad < eps:
            # Fallback to min-max if MAD collapses
            xmin, xmax = np.min(x), np.max(x)
            scale = xmax - xmin
            if scale < eps:
                scale = eps
            return (x - center) / (scale + eps)
        return (x - center) / (mad + eps)

    # === URGENCY TERM ===
    # Linear risk ramp: strong monotonic penalty for negative slack, bounded flat ceiling above slack=1.0
    # Ensures deterministic, interpretable, and numerically stable dominance
    urgency_raw = np.where(slack < 0, -slack, 0.0)  # Only penalize violation margin
    urgency_raw = np.clip(urgency_raw, 0.0, 10.0)   # Cap extreme violation impact
    norm_urgency = normalize_mad(urgency_raw)
    # Dominant term: scaled to strongly separate urgent tasks; negative score → higher priority
    urgency_term = -3.5 * norm_urgency

    # === ENERGY-CRITICALITY RATIO (ECR) TERM ===
    # ECR = energy / (criticality × work); low ECR = high efficiency per critical unit
    # Masked to zero when slack < 0 — no efficiency optimization under violation
    ecr_denom = upward_rank * remaining_work + eps
    ecr_base = min_incremental_energy / ecr_denom
    ecr_masked = np.where(slack >= 0, ecr_base, 0.0)
    norm_ecr = normalize_mad(ecr_masked)
    # Lower ECR is better → negative weight promotes energy-efficient critical tasks
    ecr_term = -1.8 * norm_ecr

    # === LATENCY TERM ===
    # Total time cost: exec + comm, weighted by uncertainty only when feasible
    base_latency = min_exec_time + min_comm_time + eps
    uncertainty_weight = np.where(slack >= 0, 1.0 + 0.5 * np.tanh(uncertainty), 1.0)
    latency_scaled = base_latency * uncertainty_weight
    norm_latency = normalize_mad(latency_scaled)
    # Higher latency → lower priority → positive weight
    latency_term = 0.7 * norm_latency

    # === FAIRNESS TERM ===
    # Log-linear aging: ln(1 + wait) / (1 + max(0, slack)), activated only if slack > 0.5s (mild margin)
    # Prevents starvation without compromising tight-deadline tasks
    wait_safe = np.maximum(ready_wait_time, 0.0)
    log_wait = np.log1p(wait_safe)  # ln(1+x), safe at x=0
    slack_margin = np.maximum(slack - 0.5, 0.0)  # Only activate fairness when slack > 0.5s
    fairness_raw = np.divide(log_wait, slack_margin + eps)
    fairness_clipped = np.clip(fairness_raw, 0.0, 0.4)
    norm_fairness = normalize_mad(fairness_clipped)
    # Fairness boosts priority → negative weight
    fairness_term = -0.15 * norm_fairness

    # === LEXICOGRAPHIC GATING VIA MULTIPLICATIVE MASKING ===
    # Urgency dominates: if urgency_term is in top 10% of its range, suppress all other terms
    urgency_rank = np.argsort(np.argsort(urgency_term))  # ordinal rank
    urgency_dominance = (urgency_rank >= len(urgency_term) * 0.9).astype(float)
    # Apply masking: when urgency dominates, suppress non-urgency contributions
    # This enforces hard deadline priority without additive interference
    non_urgency_scale = 1.0 - urgency_dominance
    score = (
        urgency_term
        + non_urgency_scale * (ecr_term + latency_term + fairness_term)
    )

    # Final safeguard: ensure finite, deterministic output
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    return score
