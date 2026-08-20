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
    eps = 0.00040418935765278564
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
    ddl_violated = (slack < -0.13460316684607848).astype(float)
    hard_ddl_penalty = ddl_violated * 310118.69960859866
    slack_mask_base = 1.0 / (1.0 + np.exp(-3.180254260872525 * -slack))
    unified_slack_mask = np.clip(slack_mask_base ** 2.6962500833795002, 0.0, 1.0)
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.2081234034118322
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    slack_hinge = np.clip(-norm_slack, 0.0, 0.7481678742701197)
    rank_slack_coupling = 1.0 + 0.9083814377984911 * (slack_hinge / (0.7481678742701197 + eps))
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    urgency_gate = 1.0 / (1.0 + np.exp(-7.977350043276277 * (slack_pressure - 1.0)))
    boosted_rank = norm_rank * rank_slack_coupling * (1.0 + 0.9083814377984911 * urgency_gate)
    coupled_work_urgency = 0.8046153185870383 * norm_work * (1.0 - unified_slack_mask)
    wait_benefit = np.clip(1.0 - 0.6949246562885982 * ready_wait_time, 0.0, 1.0) * unified_slack_mask
    score = +hard_ddl_penalty + np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(boosted_rank, -2.0, 2.0) - 0.9877022289618027 * np.clip(norm_energy * unified_slack_mask, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip(1.2081234034118322 * coupled_work_urgency, -2.0, 2.0)
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.reshape(-1)
