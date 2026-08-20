import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule: merges Parent 2's robust hinge & joint gate with Parent 1's synthetic load proxy;
       introduces starvation_saturation_offset for sharper low-wait discrimination;
       replaces redundant duration_robustness with duration_risk_coupling_strength to unify congestion modeling;
       retains validated softplus starvation, clipped linear urgency, and unified joint slack–uncertainty gating;
       eliminates all unbounded ops and ensures strict [-2,2] clipping per term to guarantee stability under CMA-ES."""
    eps = 0.01210960043640748
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
    slack_feasibility = 1.0 / (1.0 + np.exp(-4.4280449760552685 * slack))
    uncert_feasibility = 1.0 / (1.0 + np.exp(4.4280449760552685 * (norm_uncert - 0.21209408582343336)))
    joint_gate = slack_feasibility * uncert_feasibility
    ddl_breach = (slack <= 0.0).astype(float)
    critical_path_leverage = norm_rank * norm_work * ddl_breach
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 2.617933318552489
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    slack_hinge = np.clip(-norm_slack, 0.0, 1.0)
    rank_slack_coupling = 1.0 + 1.0395402041079977 * slack_hinge
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-1.174916089995666 * (slack_pressure - 1.0)))
    boosted_rank = norm_rank * rank_slack_coupling * (1.0 + 1.0395402041079977 * rank_gate)
    wait_benefit = np.log1p(np.exp(-0.8387331877888136 * (ready_wait_time + 0.313370820775956)))
    congestion_proxy = 0.2645717020683109 * norm_duration * norm_uncert * joint_gate
    energy_uncert_penalty = norm_energy * norm_uncert * joint_gate
    energy_slack_penalty = norm_energy * slack_pressure * joint_gate
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(critical_path_leverage, -2.0, 2.0) - np.clip(boosted_rank, -2.0, 2.0) - 0.64027540726161 * np.clip(norm_energy * joint_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + 0.07248998456062561 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 2.617933318552489 * np.clip(energy_slack_penalty, -2.0, 2.0) + 0.7596685203948892 * np.clip(norm_work, -2.0, 2.0) + np.clip(congestion_proxy, -2.0, 2.0)
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.reshape(-1)
