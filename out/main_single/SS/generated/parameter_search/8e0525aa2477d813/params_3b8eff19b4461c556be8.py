import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining robustness from Parent 1 and strict DDL enforcement from Parent 2:
       - Uses median-MAD normalization (Parent 2) for stability, but adds *urgency_sharpening_factor* (from Parent 1)
         applied to MAD-normalized slack for stronger primary urgency signal.
       - Keeps hard_ddl_violation_penalty + slack_sensitivity_center for strict DDL-first behavior.
       - Introduces *remaining_work_weight_under_pressure*: activates downstream work preference *only* under slack<=0,
         replacing generic work weighting — avoids diluting urgency with irrelevant large-work tasks.
       - Removes all quantile-based gating and uncertainty interaction terms (overfitting per analysis);
         instead uses unified slack-conditioned logic with sharpened urgency and binary pressure activation.
       - All intermediate terms clipped to [-2,2] for bounded AST depth and gradient-friendly tuning."""
    eps = 6.905517139785023e-05
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
        spread = np.maximum(mad, eps)
        return (x - med) / spread
    norm_slack = median_mad_normalize(slack)
    norm_energy = median_mad_normalize(min_incremental_energy)
    norm_rank = median_mad_normalize(upward_rank)
    norm_work = median_mad_normalize(remaining_work)
    slack_med = np.median(slack)
    slack_mad = np.median(np.abs(slack - slack_med))
    slack_spread = np.maximum(slack_mad, eps)
    norm_slack_mad = (slack - slack_med) / slack_spread
    sharpened_urgency = 0.7859387233232523 * norm_slack_mad
    ddl_violated = (slack < -0.18866691593409918).astype(float)
    hard_ddl_penalty = ddl_violated * 107173.93550196078
    slack_feasibility = 1.0 / (1.0 + np.exp(-7.997947347072511 * slack))
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.442035867474908
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    slack_hinge = np.clip(-norm_slack, 0.0, 0.3478492336520244)
    rank_slack_coupling = 1.0 + 1.480807795774937 * (slack_hinge / (0.3478492336520244 + eps))
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    urgency_gate = 1.0 / (1.0 + np.exp(-6.195388752488236 * (slack_pressure - 1.0)))
    boosted_rank = norm_rank * rank_slack_coupling * (1.0 + 1.480807795774937 * urgency_gate)
    wait_benefit = np.clip(1.0 - 0.3447911518102541 * ready_wait_time, 0.0, 1.0)
    ddl_pressure = (slack <= 0.0).astype(float)
    work_under_pressure = 1.3130963084806904 * norm_work * ddl_pressure
    score = +hard_ddl_penalty + np.clip(norm_slack_penalty, -2.0, 2.0) + np.clip(sharpened_urgency, -2.0, 2.0) - np.clip(boosted_rank, -2.0, 2.0) - 0.8771447592086514 * np.clip(norm_energy * slack_feasibility, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) - np.clip(work_under_pressure, -2.0, 2.0)
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.reshape(-1)
