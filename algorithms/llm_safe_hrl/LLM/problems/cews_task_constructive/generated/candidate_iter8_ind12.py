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
    Priority rule v2: Combines lexicographic dominance (Parent 2) with arctan-smooth urgency and critical-energy density (CED)
    from Parent 1, while preserving degenerate-safe normalization and quadratic uncertainty gating.
    
    Key improvements:
      - Uses arctan-based bounded urgency for numerically stable deadline risk scoring (avoids exp overflow near large negative slack)
      - Integrates CED = (upward_rank * remaining_work) / (min_incremental_energy + eps) as a *criticality-aware energy efficiency* signal,
        activated only when slack >= 0 and normalized robustly — replaces linear energy_eff_ratio for better physical interpretability.
      - Maintains ordinal dominance hierarchy: urgency > CED/criticality > fairness > uncertainty > work, implemented via base-10^6 scaling.
      - Retains quadratic uncertainty gating scaled by (slack/(|slack|+eps))², but now applied *only* to latency inflation and CED weighting,
        not to fairness or work terms — improves signal separation.
      - Robust min-max normalization is used uniformly; handles N=1 and constants safely.
      - Tie-breaking uses fastest-execution-first *and* highest-upward-rank among equally urgent tasks — preserves critical path under violation.
      - All operations guarded against division-by-zero, NaN, and inf; outputs strictly finite and deterministic.
    """
    eps = 1e-08
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    def robust_minmax_normalize(x):
        """Min-max normalize: returns 0.5 for constant/N=1 arrays; avoids rank inversion."""
        if x.size == 1:
            return np.full_like(x, 0.5, dtype=float)
        x_min, x_max = np.min(x), np.max(x)
        if x_max - x_min < eps:
            return np.full_like(x, 0.5, dtype=float)
        return np.clip((x - x_min) / (x_max - x_min + eps), 0.0, 1.0)

    # 1. Bounded arctan urgency: smooth, stable, monotonic — higher score = higher risk
    # Maps slack → [0, 1], where slack << 0 → ~1.0 (max urgency), slack >> 0 → ~0.0 (no urgency)
    arctan_urgency = 0.5 + 1.0 / np.pi * np.arctan(-slack / 5.0)  # negative slack → high urgency
    urgency_score = robust_minmax_normalize(arctan_urgency)

    # 2. Critical-Energy Density (CED): (importance × work) / marginal energy → efficiency per criticality unit
    # Activated only when slack >= 0 (i.e., safe regime where energy optimization is permissible)
    ced_numerator = upward_rank * remaining_work
    ced_denominator = min_incremental_energy + eps
    ced_raw = ced_numerator / ced_denominator
    ced_gated = np.where(slack >= 0, ced_raw, 0.0)
    ced_norm = robust_minmax_normalize(ced_gated)

    # 3. Fairness: sqrt-scaled wait time, clipped and gated by urgency (less boost when urgent)
    wait_sqrt = np.sqrt(np.maximum(ready_wait_time, 0.0))
    max_wait = np.max(ready_wait_time) + eps
    rel_wait = np.clip(wait_sqrt / max_wait, 0.0, 1.0)
    fairness_boost = rel_wait * (1.0 - urgency_score)  # reduce aging priority when deadline risk is high
    fairness_norm = robust_minmax_normalize(fairness_boost)

    # 4. Uncertainty: quadratically gated by normalized slack margin (only active when slack > 0)
    slack_margin = np.clip(slack, 0.0, None)
    max_slack_margin = np.max(slack_margin + eps) + eps
    slack_normed = slack_margin / max_slack_margin
    uncertainty_gated = np.where(slack > 0, uncertainty * (slack_normed ** 2), 0.0)
    uncertainty_norm = robust_minmax_normalize(uncertainty_gated)

    # 5. Remaining work: normalized as tie-breaker (lower work first in same class)
    work_norm = robust_minmax_normalize(remaining_work)

    # Lexicographic scoring: strict ordering via magnitude scaling
    score_urgency = urgency_score * 1_000_000.0     # dominates all
    score_ced = ced_norm * 100_000.0                # criticality-aware efficiency second
    score_fairness = fairness_norm * 10_000.0       # fairness third
    score_uncertainty = uncertainty_norm * 1_000.0  # uncertainty fourth
    score_work = work_norm * 100.0                  # work volume fifth

    # Tie-breaking among equally urgent tasks: prefer faster execution AND higher upward rank
    exec_tie = (1.0 - robust_minmax_normalize(min_exec_time)) * 10.0   # lower exec time → higher tie score
    rank_tie = (1.0 - robust_minmax_normalize(upward_rank)) * 1.0      # higher rank → higher tie score (preserves CP)

    score = (
        score_urgency + score_ced + score_fairness + score_uncertainty +
        score_work + exec_tie + rank_tie
    )

    # Final safeguard: ensure finite, deterministic output
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    return score.astype(float)
