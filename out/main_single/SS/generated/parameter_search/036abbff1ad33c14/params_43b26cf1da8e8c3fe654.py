import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with strict hard-deadline enforcement:
       - Introduces *finite but dominant* hard_ddl_violation_penalty applied only when slack < slack_sensitivity_center,
         replacing infinite scores (which break gradient-based tuning) while preserving DDL-first dominance.
       - Restores slack_pressure_gate_steepness (dropped in v1) for sharper, more responsive critical-path activation.
       - Removes exec_comm_balance and energy_uncertainty_interaction per reflection — both were overfitting noise;
         instead uses unified slack-conditioned penalty structure with single sensitivity center.
       - Keeps robust median-MAD normalization and bounded linear starvation from v2 baseline.
       - All intermediate terms clipped to [-2,2] and final score safeguarded with nan_to_num using machine precision bounds."""
    eps = 1.9310517465243957e-06
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
    norm_rank = median_mad_normalize(upward_rank)
    norm_work = median_mad_normalize(remaining_work)
    ddl_violated = (slack < -0.168623024940331).astype(float)
    hard_ddl_penalty = ddl_violated * 1000.0
    slack_feasibility = 1.0 / (1.0 + np.exp(-9.924972625837448 * slack))
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 2.637908950734912
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    slack_hinge = np.clip(-norm_slack, 0.0, 0.8883669261820769)
    rank_slack_coupling = 1.0 + 0.6595259022367486 * (slack_hinge / (0.8883669261820769 + eps))
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    urgency_gate = 1.0 / (1.0 + np.exp(-6.796591336288548 * (slack_pressure - 1.0)))
    boosted_rank = norm_rank * rank_slack_coupling * (1.0 + 0.6595259022367486 * urgency_gate)
    wait_benefit = np.clip(1.0 - 0.808345981438863 * ready_wait_time, 0.0, 1.0)
    score = +hard_ddl_penalty + np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(boosted_rank, -2.0, 2.0) - 0.81747958220644 * np.clip(norm_energy * slack_feasibility, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + 2.637908950734912 * np.clip(norm_work * ddl_violated, -2.0, 2.0)
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.reshape(-1)
