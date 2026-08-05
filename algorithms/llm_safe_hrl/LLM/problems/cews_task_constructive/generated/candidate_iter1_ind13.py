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
      to avoid unbounded growth while ensuring starvation prevention.
    - Combines communication & execution into *latency pressure*: (exec+comm)/slack
      when slack>0; large penalty when slack<=0.
    - All operations are finite, epsilon-guarded, and deterministic.
    """
    eps = 1e-8

    # Safe array conversion without mutation
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
        q25, q75 = np.percentile(x, [25, 75], method='midpoint') if x.size > 2 else (np.median(x), np.median(x))
        iqr = q75 - q25 + eps
        center = np.median(x) + eps
        return (x - center) / iqr

    # --- Deadline urgency: exponential gating on slack ---
    # For slack <= 0: exp(-slack) grows fast → strongly penalize (i.e., prioritize)
    # For slack > 0: capped at 1.0 to avoid suppressing non-urgent tasks
    deadline_urgency = np.where(
        slack <= 0,
        np.exp(-slack + eps),  # shift to avoid exp(0) ambiguity; ensures >1 for slack<0
        np.ones_like(slack)
    )

    # --- Latency pressure: (exec + comm) normalized by effective slack ---
    # When slack <= 0: use fixed large penalty (avoid division by zero/negative)
    # When slack > 0: ratio reflects how much latency eats into buffer
    exec_comm_sum = min_exec_time + min_comm_time
    latency_pressure = np.where(
        slack <= 0,
        10.0 * np.ones_like(slack),  # strong fixed penalty for overdue risk
        np.clip(exec_comm_sum / (slack + eps), 0.0, 10.0)  # bounded [0,10]
    )

    # --- Energy-normalized criticality: higher upward_rank matters more on low-energy VMs ---
    # Use reciprocal energy (with floor) to avoid blowup; scaled robustly
    inv_energy = 1.0 / (min_incremental_energy + eps)
    energy_norm_crit = upward_rank * inv_energy
    energy_norm_crit_scaled = robust_scale(energy_norm_crit)

    # --- Waiting saturation: arctan-based soft cap to prevent domination ---
    wait_saturation = np.arctan(ready_wait_time / (np.mean(ready_wait_time + eps) + eps))

    # --- Uncertainty-adjusted risk: amplify uncertainty only when slack is tight ---
    # Modulate uncertainty by slack proximity: higher weight near deadline
    slack_distance = np.abs(slack) / (np.mean(np.abs(slack) + eps) + eps)
    risk_modulated_uncert = uncertainty * (1.0 - np.tanh(slack_distance))  # ↑ when slack near 0

    # --- Remaining work: normalize but retain sign — larger work implies more downstream impact ---
    work_scaled = robust_scale(remaining_work)

    # --- Assemble score: smaller = better ---
    # Prioritize urgency first (via multiplicative gating), then balance others additively
    score = (
        # Dominant deadline term: urgency gates overall priority
        3.0 * deadline_urgency
        # Latency pressure adds cost for consuming slack
        + 1.5 * latency_pressure
        # Favor critical tasks on energy-efficient VMs
        - 1.2 * energy_norm_crit_scaled
        # Encourage progress on large remaining work
        + 0.8 * work_scaled
        # Mild anti-starvation via saturated wait time
        - 0.6 * wait_saturation
        # Risk modulation only when deadline is tight
        + 0.9 * robust_scale(risk_modulated_uncert)
        # Minimal direct energy penalty (since energy-normalized criticality already covers it)
        + 0.3 * robust_scale(min_incremental_energy)
    )

    # Final numeric safeguard
    return np.nan_to_num(
        score,
        nan=1e12,
        posinf=1e12,
        neginf=-1e12
    )
