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
    'Improved priority rule combining strict deadline gating (v1) with robust IQR scaling and criticality-energy efficiency (v0/v1), plus novel starvation-aware urgency calibration and uncertainty-dampened criticality.\n\nKey improvements:\n  - Strict slack ≤ 0 gating for *all* risk/urgency terms (deadline penalty, uncertainty boost, wait boost) — ensures hard DDL dominance.\n  - Criticality-energy ratio: upward_rank / (min_exec_time + min_comm_time + eps), robustly normalized — favors high-impact-per-time tasks, resisting outlier distortion.\n  - Starvation guard: ready_wait_time boost activated only when (slack ≤ 0) OR (upward_rank > median_rank AND uncertainty > 0.1), balancing fairness and relevance.\n  - Uncertainty-dampened criticality: leveraged_rank = upward_rank * (1.0 + np.clip(uncertainty, 0.0, 0.5)) * (1.0 + work_scale), enhancing critical-path focus under risk without over-amplification.\n  - Unified robust normalization using IQR fallback to MAD (not mean-abs) for superior outlier resilience.\n  - All terms finite, deterministic, and numerically protected.'
    eps = 1e-08
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    
    # Robust normalization: IQR-based, fallback to MAD if IQR near zero
    def robust_normalize(x):
        if x.size == 0:
            return np.zeros_like(x)
        q1, q3 = np.quantile(x, [0.25, 0.75], method='midpoint')
        iqr = q3 - q1
        if iqr < eps:
            mad = np.median(np.abs(x - np.median(x)))
            scale = mad if mad > eps else np.mean(np.abs(x)) + eps
        else:
            scale = iqr + eps
        return x / scale
    
    # Strict urgency gating: only activate risk-aware terms when slack <= 0
    is_deadline_risky = slack <= 0
    # Deadline penalty: large additive penalty for negative slack, smooth decay for positive slack
    deadline_penalty = np.where(
        is_deadline_risky,
        15.0 + np.abs(slack),
        np.exp(-slack / (np.maximum(np.mean(np.abs(slack[slack != 0])), eps) + eps))
    )
    
    # Criticality-energy efficiency ratio: impact per time unit
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    crit_eff_ratio = upward_rank / duration
    norm_crit_eff = robust_normalize(crit_eff_ratio)
    
    # Uncertainty-dampened & work-leveraged criticality
    work_scale = robust_normalize(remaining_work)
    # Cap uncertainty contribution to avoid explosion; enhance only under moderate risk
    unc_factor = 1.0 + np.clip(uncertainty, 0.0, 0.5)
    leveraged_rank = upward_rank * unc_factor * (1.0 + work_scale)
    norm_leveraged_rank = robust_normalize(leveraged_rank)
    
    # Energy efficiency: energy per second, capped and normalized
    energy_per_second = np.clip(min_incremental_energy / duration, -1e6, 1e6)
    norm_energy_ps = robust_normalize(energy_per_second)
    
    # Starvation guard: wait boost only if deadline-risky OR (critical AND uncertain)
    median_rank = np.median(upward_rank) if len(upward_rank) > 1 else 0.0
    wait_activation = np.logical_or(
        is_deadline_risky,
        np.logical_and(upward_rank > median_rank, uncertainty > 0.1)
    )
    norm_wait = np.where(wait_activation, robust_normalize(ready_wait_time), 0.0)
    
    # Uncertainty boost only under deadline risk
    norm_uncertainty = np.where(is_deadline_risky, robust_normalize(uncertainty), 0.0)
    
    # Final score: smaller = better
    # Prioritize deadline compliance first (high weight), then energy efficiency, criticality, and fairness
    score = (
        1.2 * deadline_penalty +           # Dominant hard-DDL enforcement
        0.25 * norm_energy_ps +            # Reward low energy-per-second
        0.15 * norm_leveraged_rank +       # Favor uncertainty-aware critical path
        0.1 * norm_crit_eff +              # Boost impact-per-time tasks
        0.05 * norm_uncertainty +          # Mild uncertainty boost when risky
        0.05 * norm_wait                   # Gentle starvation prevention
    )
    
    # Ensure finiteness and determinism
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    return score
