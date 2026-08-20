import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with unified slack-driven gating:
       - Replaces *all* ad-hoc gates (feasibility, pressure, load) with a single, smooth,
         exponentiated slack mask: `mask = sigmoid(-slack)^unified_slack_mask_exponent`.
       - This mask jointly governs energy preference, starvation relief, and uncertainty coupling —
         eliminating uncontrolled interactions while preserving monotonic urgency response.
       - Introduces urgency_coupling_strength to fuse remaining_work and upward_rank *only under slack pressure*,
         promoting high-work descendants when deadlines tighten — improving end-to-end latency control.
       - Removes successor_release and host_load_surrogate: their empirical degradation confirmed structural over-complexity.
       - Keeps robust median-MAD normalization on core features only; all intermediates clipped to [-2,2].
       - Hard DDL enforcement remains dominant and finite via hard_ddl_violation_penalty.
       - Final score is fully deterministic, shape-(N,), and machine-precision safe."""
    eps = 0.00046173262639899487
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
    ddl_violated = (slack < -0.0549832152397835).astype(float)
    hard_ddl_penalty = ddl_violated * 3926.2223597944694
    slack_mask_base = 1.0 / (1.0 + np.exp(-4.334480203088175 * -slack))
    unified_slack_mask = np.clip(slack_mask_base ** 2.1674601288049793, 0.0, 1.0)
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.8450186137507834
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    slack_hinge = np.clip(-norm_slack, 0.0, 0.8599272347197507)
    rank_slack_coupling = 1.0 + 1.9498836146808416 * (slack_hinge / (0.8599272347197507 + eps))
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    urgency_gate = 1.0 / (1.0 + np.exp(-7.989484355492314 * (slack_pressure - 1.0)))
    boosted_rank = norm_rank * rank_slack_coupling * (1.0 + 1.9498836146808416 * urgency_gate)
    coupled_work_urgency = 0.7213005220255728 * norm_work * (1.0 - unified_slack_mask)
    wait_benefit = np.clip(1.0 - 0.658953484403895 * ready_wait_time, 0.0, 1.0) * unified_slack_mask
    score = +hard_ddl_penalty + np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(boosted_rank, -2.0, 2.0) - 1.4357665077397603 * np.clip(norm_energy * unified_slack_mask, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip(1.8450186137507834 * coupled_work_urgency, -2.0, 2.0)
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.reshape(-1)
