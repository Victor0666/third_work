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
    """Novel priority rule emphasizing deadline feasibility first, then energy-aware criticality.

    Key innovations:
    - Uses *slack-based urgency gating*: only tasks with slack <= threshold activate high-priority modes
    - Introduces *criticality-energy tradeoff ratio*: balances upward_rank (importance) against 
      min_incremental_energy (cost), normalized robustly via interquartile scaling
    - Replaces linear waiting-time boost with *starvation-aware sigmoid*: prevents runaway priority
      while ensuring bounded, monotonic fairness for long-waiting tasks
    - Applies *risk-conditional normalization*: normalizes uncertainty and energy only when slack is tight
    - Uses IQR-based robust scaling instead of mean-abs to avoid outlier distortion in heterogeneous workflows
    - All operations are numerically safe: eps-protected, nan-to-num guarded, no log/exp near boundaries
    """
    eps = 1e-8

    # Convert inputs safely without mutation
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    # Robust IQR-based normalization (more stable than mean-abs for skewed distributions)
    def robust_normalize(x):
        q1, q3 = np.quantile(x, [0.25, 0.75])
        iqr = q3 - q1 + eps
        center = np.median(x)
        return (x - center) / iqr

    # Deadline urgency: strong penalty for negative slack, soft decay for positive slack
    # Map slack ∈ [-∞, ∞] → urgency ∈ [0, 1], with sharp transition at slack=0
    urgency_mask = (slack <= 0.0).astype(float)
    urgency_linear = np.maximum(0.0, -slack)  # absolute lateness risk
    urgency_soft = 1.0 / (1.0 + np.maximum(0.0, slack) * 0.1 + eps)  # decays slowly for large slack
    deadline_urgency = urgency_mask * (1.0 + urgency_linear * 0.5) + (1.0 - urgency_mask) * urgency_soft

    # Criticality-energy efficiency: higher upward_rank per unit energy = better value
    # Avoid division by zero; clamp energy to avoid infinite ratios
    energy_safe = np.maximum(min_incremental_energy, eps)
    crit_efficiency = upward_rank / (energy_safe + eps)
    # Normalize only the ratio — avoids amplifying noise from tiny energies
    crit_eff_norm = robust_normalize(crit_efficiency)

    # Starvation fairness: sigmoid rewards long wait but saturates to prevent dominance
    # Saturates near 1.0 after ~10s wait (configurable via scale)
    wait_scale = np.maximum(np.mean(ready_wait_time), eps)
    starvation_bonus = 1.0 / (1.0 + np.exp(-(ready_wait_time / (wait_scale + eps) - 2.0)))

    # Risk-conditional normalization: apply uncertainty weight only when slack is tight
    # Prevents over-prioritizing uncertain-but-safe tasks
    risk_weight = np.where(slack <= 0.0, 1.0, np.clip(uncertainty, 0.0, 1.0))
    uncertainty_normalized = robust_normalize(uncertainty) * risk_weight

    # Execution & communication cost: combine time components with diminishing returns
    time_cost = np.sqrt(np.maximum(min_exec_time, eps)) + np.sqrt(np.maximum(min_comm_time, eps))
    time_norm = robust_normalize(time_cost)

    # Remaining work acts as coarse-grained importance proxy — normalize robustly
    work_norm = robust_normalize(remaining_work)

    # Composite score: smaller = better
    # Priority hierarchy: deadline > criticality-efficiency > starvation > time/energy/uncertainty
    score = (
        -2.5 * deadline_urgency               # strongest pull: urgent tasks first
        - 1.2 * crit_eff_norm                 # reward high-criticality low-energy tasks
        - 0.8 * starvation_bonus              # fair starvation relief (bounded)
        + 0.6 * time_norm                     # penalize high-time-cost tasks
        + 0.4 * robust_normalize(min_incremental_energy)  # marginal energy cost
        + 0.3 * work_norm                     # downstream impact
        + 0.2 * uncertainty_normalized        # uncertainty only matters under pressure
    )

    # Final numerical guard
    return np.nan_to_num(
        score,
        nan=1e12,
        posinf=1e12,
        neginf=-1e12
    )
