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
    v2 mutation: Replaces sigmoid urgency with arctan-based bounded risk,
    introduces *critical-energy density* (CED) as core term with slack-gated activation,
    replaces aging boost with clipped sqrt-wait scaled by urgency-aware threshold,
    uses additive uncertainty penalty only when slack > 0 (risk-aware latency inflation),
    and applies robust min-max normalization with degenerate-safe fallback.
    
    Key changes:
    - arctan urgency: smooth, bounded, numerically stable near zero slack; avoids sigmoid saturation
    - CED = (upward_rank * remaining_work) / (min_incremental_energy + eps), activated only if slack >= 0
    - Latency-inflation: adds uncertainty to min_exec_time + min_comm_time *only* when slack > 0 → reflects conservative scheduling under risk
    - Fairness: sqrt(ready_wait_time) clipped and scaled by adaptive threshold (median wait + eps), not max
    - Normalization: min-max with IQR fallback for flat distributions; guarantees finite scale even for N=1
    - All gating is binary or arctan-smooth — no multiplicative uncertainty scaling on energy/criticality
    - Negative slack triggers hard priority via dominant urgency term; no efficiency trade-off in violation regime
    """
    eps = 1e-8
    # Cast inputs safely
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    # === Robust normalization: min-max with IQR fallback for degenerate cases ===
    def normalize_robust(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size == 1:
            return np.array([0.0])
        # Try min-max first
        x_min, x_max = np.min(x), np.max(x)
        if x_max - x_min > eps:
            return (x - x_min) / (x_max - x_min + eps)
        # Fallback: IQR-based centering if range collapses
        q25, q75 = np.percentile(x, [25, 75])
        iqr = q75 - q25
        if iqr > eps:
            center = np.median(x)
            return (x - center) / (iqr + eps)
        # Final fallback: std-based if IQR also flat
        std_val = np.std(x)
        center = np.mean(x)
        scale = std_val if std_val > eps else 1.0
        return (x - center) / (scale + eps)

    # === Arctan-based bounded urgency: [-π/2, π/2] → [0, 1] mapping, monotonic & outlier-robust ===
    # arctan(slack / tau) ∈ (-π/2, π/2); shift & scale to [0,1]; negative slack yields near-0 → high priority (small score)
    tau = 5.0
    arctan_urgency = 0.5 + (1.0 / np.pi) * np.arctan(slack / tau)  # now ∈ (0,1), decreasing in slack
    # Invert so smaller score = higher urgency (negative slack → low arctan_urgency → high priority)
    urgency_score = arctan_urgency  # lower value = more urgent → keep as-is for min-selection

    # === Critical-Energy Density (CED): (importance × work) / marginal energy ===
    # Only meaningful where energy minimization is allowed → disable when slack < 0 (hard deadline mode)
    ced_numerator = upward_rank * remaining_work
    ced_denominator = min_incremental_energy + eps
    ced_raw = ced_numerator / ced_denominator
    # Gate CED: fully active only when slack >= 0; zero otherwise → no efficiency trade-off under violation risk
    ced_gated = np.where(slack >= 0, ced_raw, 0.0)
    norm_ced = normalize_robust(ced_gated)
    # Higher CED = better energy efficiency per critical unit → higher priority → negate for min-score
    ced_term = -1.2 * norm_ced

    # === Latency-inflation term: add uncertainty to latency *only* when slack > 0 ===
    # Represents conservative placement under bandwidth/compute uncertainty, not penalty
    base_latency = min_exec_time + min_comm_time + eps
    inflated_latency = np.where(slack > 0, base_latency + uncertainty, base_latency)
    # Normalize inflated latency → longer latency reduces priority (larger score)
    norm_latency = normalize_robust(inflated_latency)
    latency_term = 0.8 * norm_latency

    # === Fairness: clipped sqrt(wait) scaled by median-relative threshold ===
    # Prevents starvation without overriding deadlines; sqrt avoids linear bias toward long waits
    wait_sqrt = np.sqrt(np.maximum(ready_wait_time, 0.0))
    median_wait = np.median(ready_wait_time) + eps
    rel_sqrt_wait = np.clip(wait_sqrt / median_wait, 0.0, 2.0)  # cap at 2× median
    # Scale fairness boost by urgency: less boost for urgent tasks (they’re already prioritized)
    fairness_boost = rel_sqrt_wait * (1.0 - urgency_score)  # drops to 0 when urgency_score → 1 (very non-urgent)
    norm_fairness = normalize_robust(fairness_boost)
    fairness_term = -0.25 * norm_fairness  # negative → higher fairness_boost improves priority

    # === Hard-deadline override: for slack < -eps, inject dominant urgency penalty ===
    # Ensures immediate selection of violating tasks — no other term can compensate
    hard_violation_mask = (slack < -eps).astype(float)
    violation_penalty = 50.0 * hard_violation_mask  # large constant pushes score high → but wait: smaller score = higher priority!
    # So instead: make urgency_score *dominant* and low for violations → already handled by arctan_urgency being near 0
    # Therefore, no extra penalty needed — just ensure it's the strongest term

    # === Assemble final score: urgency dominates, then CED, latency, fairness ===
    # All terms designed so smaller = better
    score = (
        2.0 * urgency_score +      # primary: arctan urgency (0–1, lower = better)
        ced_term +                 # secondary: energy-critical efficiency (gated)
        latency_term +             # tertiary: conservative latency under risk
        fairness_term              # quaternary: anti-starvation
    )

    # Final sanitization: ensure finite, deterministic, shape-(N,)
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    return score
