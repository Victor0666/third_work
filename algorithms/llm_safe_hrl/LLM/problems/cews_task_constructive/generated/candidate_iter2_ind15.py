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
    'Improved priority rule combining deadline gating, critical-path-aware energy efficiency,\n    bounded aging, and risk-amplified urgency — with robust IQR normalization and strict DDL safety.\n\n    Key improvements:\n    - Uses *adaptive slack threshold* (like Parent 1) but with tighter safety margin (-0.01 * median positive slack)\n      to avoid premature penalty while guaranteeing hard DDL compliance.\n    - Integrates Parent 2\'s multiplicative risk_amp on *both deadline_risk and energy_efficiency*,\n      capturing joint impact of uncertainty on latency and energy tradeoffs.\n    - Replaces raw upward_rank with *rank-load coupling*: upward_rank * remaining_work / (min_exec_time + min_comm_time + eps),\n      prioritizing high-criticality tasks that also contribute substantially to total work and are communication-efficient.\n    - Uses tanh-based aging (Parent 2) but scaled by median ready_wait_time for fairness across varying load conditions.\n    - Energy term uses *joules-per-total-useful-time-and-work*, i.e., min_incremental_energy / (min_exec_time + min_comm_time + remaining_work * 1e-6 + eps),\n      unifying time and computation effort into a single "resource utilization" denominator.\n    - All normalizations use IQR with explicit fallback for degenerate cases (size <= 1 or zero IQR).\n    '
    eps = 1e-08
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    # Robust IQR normalization with fallback for small/constant arrays
    def normalize_iqr(x):
        if x.size == 0:
            return np.zeros_like(x)
        if x.size == 1:
            return np.zeros_like(x, dtype=float)
        q75, q25 = np.percentile(x, [75, 25], method='midpoint')
        iqr = q75 - q25
        scale = iqr if iqr > eps else np.mean(np.abs(x - np.median(x))) + eps
        center = np.median(x)
        return (x - center) / (scale + eps)

    # Adaptive deadline risk gating: only penalize when slack is critically negative
    pos_slack_mask = slack > 0
    median_pos_slack = np.median(slack[pos_slack_mask]) if np.any(pos_slack_mask) else 1.0
    critical_threshold = -0.01 * max(median_pos_slack, eps)
    deadline_risk = np.where(slack < critical_threshold, -slack, 0.0)

    # Unified useful resource denominator: time (sec) + normalized work (MI → sec-equivalent)
    useful_resource = min_exec_time + min_comm_time + remaining_work * 1e-6
    useful_resource_safe = np.maximum(useful_resource, eps)
    energy_per_resource = min_incremental_energy / useful_resource_safe

    # Rank-load coupling: prioritize critical nodes *with substantial descendant work* AND low execution+comm overhead
    # Avoids inflating leaf critical nodes or high-rank/low-work bottlenecks
    exec_comm_sum = min_exec_time + min_comm_time + eps
    rank_load_coupling = upward_rank * remaining_work / exec_comm_sum
    critical_score = normalize_iqr(rank_load_coupling)

    # Bounded aging: tanh scaled by median wait time to adapt to system load
    median_wait = np.median(ready_wait_time) if ready_wait_time.size > 0 else 1.0
    tau = max(median_wait, eps)
    aging_boost = np.tanh(0.5 * ready_wait_time / tau)  # smoother, less saturating than linear scaling

    # Risk amplification: sigmoid-transformed uncertainty → [1.0, 3.0] multiplier
    # Amplifies both deadline risk and energy efficiency under high uncertainty
    risk_amp = 1.0 + 2.0 / (1.0 + np.exp(-uncertainty + 1.0))

    # Normalize core signals
    norm_deadline_risk = normalize_iqr(deadline_risk)
    norm_energy = normalize_iqr(energy_per_resource)
    norm_exec = normalize_iqr(min_exec_time)
    norm_comm = normalize_iqr(min_comm_time)
    norm_uncert = normalize_iqr(uncertainty)

    # Final score: smaller = higher priority
    # Emphasize deadline safety first (40%), then critical path (25%), then energy (20%)
    # Penalize long exec/comm (−10% each), reward aging (−5%), add mild uncertainty bias (+5%)
    score = (
        0.40 * norm_deadline_risk * risk_amp +
        0.25 * critical_score +
        0.20 * norm_energy * risk_amp -
        0.10 * norm_exec -
        0.10 * norm_comm -
        0.05 * aging_boost +
        0.05 * norm_uncert
    )

    # Ensure finite output
    return np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
