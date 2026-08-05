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
    Hybrid priority rule combining deadline-hardness, energy-criticality alignment,
    risk-aware latency pressure, and starvation-robust waiting saturation.
    
    Key improvements:
    - Uses *slack-gated exponential urgency*: exp(max(0, -slack)) — stable, no overflow, zero impact for positive slack
    - Introduces *energy-normalized criticality* (upward_rank / (min_incremental_energy + eps)) — prioritizes high-impact/low-energy assignments
    - Combines exec+comm into *latency pressure*, but only penalizes when slack <= 0; otherwise uses normalized intrinsic latency to avoid biasing short tasks
    - Applies *uncertainty-modulated energy penalty*: multiplies energy by (1 + tanh(uncertainty)) to amplify risk without explosion
    - Uses *arctan-saturated wait time* for fairness, scaled relative to task's own latency budget (exec+comm+eps) to prevent starvation of long tasks
    - All features scaled per-feature via robust median/IQR to handle outliers and heterogeneous units
    - Final score strictly prioritizes DDL adherence first (high weight on urgency), then energy-criticality, then latency & fairness
    """
    eps = 1e-08
    # Ensure float arrays, no in-place modification
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    
    # Robust per-feature scaling: median/IQR with fallback for small N
    def robust_scale(x):
        if x.size == 1:
            return np.zeros_like(x)
        q25, q75 = np.percentile(x, [25, 75], method='midpoint')
        iqr = q75 - q25 + eps
        med = np.median(x)
        return (x - med) / iqr
    
    # Deadline urgency: exponential boost only for negative slack; zero effect when slack > 0
    # Stable: exp(max(0,-slack)) avoids overflow; max(0,-slack) is 0 when slack >= 0
    slack_gap = np.maximum(0.0, -slack)
    deadline_urgency = np.exp(slack_gap)  # >= 1.0, grows sharply only under deadline pressure
    
    # Latency pressure: penalize only under tight/missed deadlines; otherwise use intrinsic latency as baseline
    exec_comm_sum = min_exec_time + min_comm_time
    latency_pressure = np.where(
        slack <= 0,
        np.clip(exec_comm_sum / (np.abs(slack) + eps), 0.0, 15.0),
        robust_scale(exec_comm_sum)  # baseline fairness for slack-rich tasks
    )
    
    # Energy-normalized criticality: higher upward_rank + lower energy → higher priority (so we minimize score)
    # Invert energy: prefer low energy → use 1/(energy+eps); multiply by upward_rank
    inv_energy = 1.0 / (min_incremental_energy + eps)
    energy_norm_crit = upward_rank * inv_energy
    
    # Uncertainty-modulated energy: penalize high-uncertainty low-energy assignments
    risk_factor = 1.0 + np.tanh(uncertainty)  # bounded in [1.0, 2.0]
    risk_weighted_energy = min_incremental_energy * risk_factor
    
    # Wait saturation: arctan of normalized wait ratio prevents starvation while respecting task scale
    wait_ratio = ready_wait_time / (exec_comm_sum + eps)
    wait_saturation = np.arctan(wait_ratio)  # bounded in [0, pi/2]
    
    # Work importance: normalize remaining_work to guide early scheduling of heavy subgraphs
    work_scaled = robust_scale(remaining_work)
    
    # Assemble score: smaller = better
    # High weight on deadline urgency (hard constraint), then energy-criticality (efficiency under feasibility),
    # moderate weights on latency pressure and work, negative weight on wait_saturation (favor older tasks)
    score = (
        4.0 * robust_scale(deadline_urgency) +
        1.8 * robust_scale(latency_pressure) -
        1.5 * robust_scale(energy_norm_crit) +
        0.9 * robust_scale(work_scaled) -
        0.7 * wait_saturation +
        0.6 * robust_scale(risk_weighted_energy)
    )
    
    # Ensure finite output: replace NaN/inf with large finite defaults
    return np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
