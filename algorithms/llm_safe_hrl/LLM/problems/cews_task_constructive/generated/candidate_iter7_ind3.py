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
    introduces *critical-energy density* (upward_rank * remaining_work / (min_incremental_energy + eps)),
    uses clipped sqrt(wait) for aging (not relative), applies uncertainty only as additive latency inflation
    when slack > 0, and enforces strict DDL gating via zeroing non-urgent criticality.
    
    Key changes:
    - arctan(slack/tau) for smooth, bounded, numerically stable urgency (no exp overflow)
    - Critical-energy density: prioritizes tasks delivering high critical path value per joule
    - DDL-gated criticality: set to zero when slack <= 0 → forces urgent tasks to ignore energy tradeoffs
    - Aging: sqrt-clipped wait time scaled by urgency mask, not relative → avoids bias in sparse ready sets
    - Uncertainty: added to latency_cost only when slack > 0 → avoids punishing already-at-risk tasks
    - Robust normalization: IQR+fallback, but applied *after* DDL gating & masking to preserve hierarchy
    - All terms designed so lower score = higher priority; no unbounded ops, no division by raw zero
    """
    eps = 1e-8
    # Cast and sanitize inputs
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    # --- 1. Bounded, monotonic urgency: arctan(slack/tau) ∈ (-π/2, π/2), then shift/scale to [0,1]
    tau = 5.0
    urgency_raw = np.arctan(slack / tau)  # smooth, finite, odd function: negative slack → negative urgency_raw
    # Map to [0,1] where 1 = most urgent (negative slack), 0 = least urgent (large positive slack)
    urgency = 0.5 - urgency_raw / np.pi  # now 0.5 at slack=0, >0.5 when slack<0, <0.5 when slack>0

    # --- 2. DDL-gated critical-energy density: only active when slack > 0; zero otherwise
    # This enforces: when deadline is violated/imminent, energy efficiency is secondary
    critical_energy_density = (
        upward_rank * remaining_work / (min_incremental_energy + eps)
    )
    gated_critical_energy = np.where(slack > 0, critical_energy_density, 0.0)

    # --- 3. Latency cost: exec + comm + (uncertainty only if slack > 0)
    latency_cost = min_exec_time + min_comm_time + eps
    latency_cost_with_uncert = np.where(
        slack > 0,
        latency_cost + np.clip(uncertainty, 0, np.percentile(uncertainty, 90) + eps),
        latency_cost
    )

    # --- 4. Aging term: sqrt-clipped wait time, scaled by urgency (not relative) → prevents starvation
    # Clipped sqrt ensures sublinear growth and avoids blowup on large wait times
    aging_base = np.sqrt(np.clip(ready_wait_time, 0, 3600.0) + eps)  # cap at 1h → ~60s max contribution
    aging_term = urgency * aging_base  # higher urgency amplifies aging boost

    # --- 5. Robust adaptive normalization (IQR fallback) — applied per term *before* weighting
    def robust_normalize(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size == 1:
            return np.array([0.0])
        q25, q75 = np.percentile(x, [25, 75])
        iqr = q75 - q25
        if iqr > eps:
            scale = iqr + eps
            center = np.median(x)
        else:
            std_val = np.std(x)
            scale = std_val if std_val > eps else 1.0
            center = np.mean(x)
        return (x - center) / (scale + eps)

    # Normalize each component separately
    norm_urgency = robust_normalize(urgency)
    norm_critical_energy = robust_normalize(gated_critical_energy)
    norm_latency_cost = robust_normalize(latency_cost_with_uncert)
    norm_aging = robust_normalize(aging_term)

    # --- 6. Assemble priority score: smaller = better
    # Urgency dominates: strong negative weight (high priority = low score)
    # Critical-energy density: positive weight → high density = high priority → negative contribution
    # Latency cost: positive weight → high cost = low priority → positive contribution
    # Aging: positive weight → high aging = high priority → negative contribution
    score = (
        -3.0 * norm_urgency           # primary deadline driver
        -1.2 * norm_critical_energy   # reward energy-efficient critical work *only* when slack > 0
        +0.8 * norm_latency_cost      # penalize high-latency tasks (less urgent ones)
        -0.4 * norm_aging             # prevent starvation without overriding deadlines
    )

    # Final safeguard: ensure finite, deterministic output
    score = np.nan_to_num(
        score,
        nan=1e9,
        posinf=1e9,
        neginf=-1e9
    )

    return score
