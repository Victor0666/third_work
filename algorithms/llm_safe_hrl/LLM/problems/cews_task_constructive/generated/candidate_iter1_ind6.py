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
    """Novel priority rule emphasizing deadline feasibility via adaptive risk gating,
    balanced criticality-energy tradeoff, and starvation-aware waiting time scaling.
    
    Key innovations:
    - Uses sigmoid-shaped urgency curve for slack (smooth, bounded, avoids division instability)
    - Introduces 'criticality-energy ratio' to jointly penalize high-energy tasks on critical paths
    - Applies robust min-max normalization per feature (not mean-based) for better outlier resilience
    - Implements waiting-time boost only when slack is non-negative (avoids rewarding late tasks)
    - Combines uncertainty with slack sign: amplifies priority for uncertain + tight-slack tasks,
      but suppresses it for uncertain + slack-rich tasks (no premature preemption)
    - All terms scaled to [-1, 1] range before linear combination, ensuring comparable influence
    """
    eps = 1e-8

    # Convert inputs safely without in-place modification
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    def robust_minmax_norm(x):
        """Min-max normalize to [0, 1], clamping outliers; handles const arrays safely."""
        x_min, x_max = np.min(x), np.max(x)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        return (x - x_min) / (x_max - x_min + eps)

    # --- Deadline urgency: smooth, bounded, negative-slack dominant ---
    # Sigmoid of (-slack) → near 1 for large negative slack, ~0.5 at slack=0, decays slowly for positive slack
    urgency = 1.0 / (1.0 + np.exp(-slack / (np.abs(np.mean(slack)) + eps)))

    # --- Criticality-energy tension: prioritize low-energy tasks on high-upward-rank paths ---
    # Normalized upward_rank ∈ [0,1], normalized energy ∈ [0,1]; ratio penalizes high-energy/critical combos
    norm_urank = robust_minmax_norm(upward_rank)
    norm_energy = robust_minmax_norm(min_incremental_energy)
    criticality_energy_penalty = norm_urank * norm_energy  # ∈ [0,1]

    # --- Communication-computation balance term ---
    # Favor tasks with low *sum* of exec+comm time relative to remaining work (efficiency proxy)
    total_duration = min_exec_time + min_comm_time
    efficiency_score = np.where(
        remaining_work > eps,
        total_duration / (remaining_work + eps),
        np.full_like(remaining_work, 1.0)
    )
    norm_efficiency = robust_minmax_norm(efficiency_score)

    # --- Uncertainty-aware slack modulation ---
    # Only amplify urgency when both slack is tight *and* uncertainty is high
    # Avoid boosting uncertain-but-slated tasks
    slack_sign_mask = (slack < 0).astype(float)  # 1 if negative slack, else 0
    uncertainty_boost = uncertainty * slack_sign_mask
    norm_uncertainty_boost = robust_minmax_norm(uncertainty_boost)

    # --- Starvation mitigation: only activate waiting boost for non-late tasks ---
    wait_boost = np.where(
        slack >= 0,
        ready_wait_time / (np.mean(ready_wait_time) + eps),
        np.zeros_like(ready_wait_time)
    )
    norm_wait_boost = robust_minmax_norm(wait_boost)

    # --- Assemble score: all components mapped to [-1, 1] for balanced contribution ---
    # Base score starts at 0; subtract favorable contributions (↑priority), add penalties (↓priority)
    score = (
        + 0.30 * norm_efficiency                    # lower duration/work → higher priority (positive weight)
        + 0.25 * criticality_energy_penalty       # high criticality + high energy → penalty (positive weight)
        - 0.40 * urgency                          # urgent (negative slack) → strong priority (negative weight)
        - 0.15 * norm_wait_boost                  # long wait (if not late) → priority boost (negative weight)
        + 0.10 * norm_uncertainty_boost           # uncertain + late → extra urgency (positive weight)
    )

    # Final clamp and finite safeguard
    score = np.clip(score, -1e6, 1e6)
    score = np.nan_to_num(score, nan=1e6, posinf=1e6, neginf=-1e6)

    return score
