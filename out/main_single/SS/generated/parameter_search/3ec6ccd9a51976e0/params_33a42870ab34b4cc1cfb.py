import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule: merges Parent 2's robust hinge & joint gate with Parent 1's synthetic load proxy;
       introduces starvation_saturation_offset for sharper low-wait discrimination;
       replaces redundant duration_robustness with duration_risk_coupling_strength to unify congestion modeling;
       retains validated softplus starvation, clipped linear urgency, and unified joint slack–uncertainty gating;
       eliminates all unbounded ops and ensures strict [-2,2] clipping per term to guarantee stability under CMA-ES."""
    eps = 0.0006066948117593376
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def median_mad_normalize(x):
        x = np.copy(x)
        if N == 1:
            med = x[0]
            mad = eps
        else:
            med = np.median(x)
            mad = np.median(np.abs(x - med))
        spread = mad if mad > eps else eps
        return (x - med) / spread
    norm_slack = median_mad_normalize(slack)
    norm_energy = median_mad_normalize(min_incremental_energy)
    norm_duration = median_mad_normalize(min_exec_time + min_comm_time)
    norm_rank = median_mad_normalize(upward_rank)
    norm_work = median_mad_normalize(remaining_work)
    norm_wait = median_mad_normalize(ready_wait_time)
    norm_uncert = median_mad_normalize(uncertainty)
    slack_feasibility = 1.0 / (1.0 + np.exp(-2.636156348728103 * slack))
    uncert_feasibility = 1.0 / (1.0 + np.exp(2.636156348728103 * (norm_uncert - 0.008430647510375347)))
    joint_gate = slack_feasibility * uncert_feasibility
    ddl_breach = (slack <= 0.0).astype(float)
    critical_path_leverage = norm_rank * norm_work * ddl_breach
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 3.188810035420363
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    slack_hinge = np.clip(-norm_slack, 0.0, 1.0)
    rank_slack_coupling = 1.0 + 0.8575132264792402 * slack_hinge
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-2.9434080216421727 * (slack_pressure - 1.0)))
    boosted_rank = norm_rank * rank_slack_coupling * (1.0 + 0.8575132264792402 * rank_gate)
    wait_benefit = np.log1p(np.exp(-0.38427584101998924 * (ready_wait_time + 0.1032040920253136)))
    congestion_proxy = 0.6079431811251484 * norm_duration * norm_uncert * joint_gate
    energy_uncert_penalty = norm_energy * norm_uncert * joint_gate
    energy_slack_penalty = norm_energy * slack_pressure * joint_gate
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(critical_path_leverage, -2.0, 2.0) - np.clip(boosted_rank, -2.0, 2.0) - 1.0029085899127652 * np.clip(norm_energy * joint_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + 0.5244042494412484 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 3.188810035420363 * np.clip(energy_slack_penalty, -2.0, 2.0) + 0.05411253359138692 * np.clip(norm_work, -2.0, 2.0) + np.clip(congestion_proxy, -2.0, 2.0)
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.reshape(-1)
