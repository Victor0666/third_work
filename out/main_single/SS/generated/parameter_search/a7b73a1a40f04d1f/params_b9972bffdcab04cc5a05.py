import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with tightened deadline focus and robust fallback:
       - Removes unused 'rank_slack_hinge_width' and redundant urgency_sharpening_factor.
       - Uses unified slack-gated criticality: sigmoid-coupled upward_rank with raw-rank fallback for stability.
       - Keeps hard_ddl_violation_penalty + slack_sensitivity_center as non-negotiable DDL enforcement core.
       - All intermediate terms clipped to [-2,2]; final score safeguarded with machine-precision bounds."""
    eps = 0.00010571124688971164
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
    ddl_violated = (slack < -0.09945713405418383).astype(float)
    hard_ddl_penalty = ddl_violated * 287780.1108310105
    slack_feasibility = 1.0 / (1.0 + np.exp(-5.428832042139868 * slack))
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 2.807635300955369
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    slack_sigmoid = 1.0 / (1.0 + np.exp(-5.754781439701457 * (slack - -0.09945713405418383)))
    rank_stability = np.where(N == 1, 0.0, np.median(np.abs(upward_rank - np.median(upward_rank))))
    use_raw_rank = (rank_stability < eps).astype(float)
    stable_rank = (1.0 - use_raw_rank) * norm_rank + use_raw_rank * (upward_rank / (np.max(np.abs(upward_rank)) + eps))
    criticality_signal = 1.2815591940558235 * slack_sigmoid * stable_rank
    wait_benefit = np.clip(1.0 - 0.524932916249516 * ready_wait_time, 0.0, 1.0)
    score = +hard_ddl_penalty + np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(criticality_signal, -2.0, 2.0) - 1.116171929654667 * np.clip(norm_energy * slack_feasibility, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + 0.5776493146852396 * np.clip(norm_work * ddl_violated, -2.0, 2.0)
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.reshape(-1)
