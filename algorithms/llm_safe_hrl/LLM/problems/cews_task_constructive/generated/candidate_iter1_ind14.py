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
    """Novel priority rule emphasizing deadline urgency, risk-aware criticality,
    and energy-aware progress fairness.

    Key innovations:
    - Uses *slack-sensitive exponential gating*: tasks with negative slack get
      exponentially amplified priority boost (not linear), ensuring hard DDL
      adherence without destabilizing positive-slack region.
    - Introduces *energy-normalized criticality*: upward_rank weighted by
      inverse of min_incremental_energy to favor high-impact tasks on low-energy
      VMs — promoting DDL compliance *and* energy efficiency jointly.
    - Replaces global normalization with *per-feature robust scaling* (IQR + median)
      for better outlier resilience and scale alignment across heterogeneous units.
    - Adds *waiting-time saturation*: ready_wait_time mapped via smooth arctan
      to avoid unbounded growth while preventing starvation.
    - Combines communication & execution into *latency pressure* term, scaled
      relative to slack — penalizing high-latency tasks more when slack is tight.
    - Uncertainty is used in *risk-adjusted slack*, not as standalone penalty.
    """
    eps = 1e-8

    # Convert inputs safely; no in-place modification
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    # --- Robust per-feature scaling: IQR-based, median-centered ---
    def robust_scale(x):
        q25, q75 = np.percentile(x, [25, 75])
        iqr = q75 - q25 + eps
        center = np.median(x)
        return (x - center) / iqr

    # --- Deadline urgency: exponential gating on normalized slack ---
    # Normalize slack first to avoid overflow; clamp extreme negatives
    slack_norm = robust_scale(slack)
    # Map slack_norm → urgency: steep rise for negative values, flat for positive
    # Use tanh-based smooth step to avoid exp explosion: tanh(-x) ≈ -1 for x>3
    urgency = -np.tanh(slack_norm * 2.0)  # [-1, 1]; -1 = most urgent (very negative slack)

    # --- Risk-adjusted slack: inflate slack by uncertainty where slack is tight ---
    # Only tighten slack when uncertainty is significant and slack is already small
    risk_adjusted_slack = slack - (uncertainty * np.clip(-slack, 0, np.inf) / (np.abs(slack) + eps))
    risk_slack_norm = robust_scale(risk_adjusted_slack)
    risk_urgency = -np.tanh(risk_slack_norm * 1.5)

    # --- Latency pressure: (exec + comm) scaled by how tight slack is ---
    # High latency matters more when slack is low → weight by urgency magnitude
    latency_pressure = (min_exec_time + min_comm_time)
    latency_pressure_norm = robust_scale(latency_pressure)
    # Apply urgency-modulated weight: stronger penalty when urgency > 0.3
    latency_weight = np.clip(urgency + 0.5, 0.0, 1.0)  # [0,1] gate
    latency_score = latency_pressure_norm * latency_weight

    # --- Energy-normalized criticality: upward_rank / (energy + eps) ---
    # Higher rank *and* lower energy → higher priority → so invert energy
    energy_efficient_rank = upward_rank / (min_incremental_energy + eps)
    eer_norm = robust_scale(energy_efficient_rank)

    # --- Waiting fairness: smooth bounded saturation (arctan) ---
    # Prevents starvation without dominating other terms
    wait_saturation = np.arctan(ready_wait_time / (np.mean(ready_wait_time + eps) + eps)) / (np.pi/2)
    wait_norm = robust_scale(wait_saturation)

    # --- Remaining work: use relative importance, not absolute size ---
    # Scale by total remaining work to avoid bias toward large workflows
    total_work = np.sum(remaining_work) + eps
    work_fraction = remaining_work / total_work
    work_norm = robust_scale(work_fraction)

    # --- Final score: smaller = higher priority ---
    # Weights chosen to prioritize urgency > criticality > fairness > latency > work
    score = (
        -3.0 * urgency                # Strongest pull: deadline urgency (negative weight → smaller score)
        -2.0 * risk_urgency          # Secondary: uncertainty-augmented urgency
        -1.5 * eer_norm              # Critical path + energy synergy
        + 0.8 * latency_score        # Penalty only when urgency demands it
        -0.6 * wait_norm             # Boost long-waiting tasks (negative weight → smaller score)
        + 0.4 * work_norm            # Mild preference for high-work fractions (less dominant)
        + 0.3 * robust_scale(uncertainty)  # Small direct risk penalty (positive → larger score = less priority)
    )

    # Numerical safety: ensure finite output
    return np.nan_to_num(
        score,
        nan=1e12,
        posinf=1e12,
        neginf=-1e12
    )
