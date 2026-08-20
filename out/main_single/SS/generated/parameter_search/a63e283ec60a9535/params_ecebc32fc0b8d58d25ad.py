import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: reinstates validated critical_path_leverage with unified DDL-feasibility gating;
       replaces dual-gated energy with single robust smooth Heaviside (ddl_gate) — eliminates instability from linear ramp;
       introduces *normalized slack alignment* via clipped, signed slack deviation to decouple urgency from outlier magnitude;
       retains median-MAD normalization for all features; removes redundant duration_robustness & wait_saturation_offset;
       enforces strict monotonic slack pressure through bounded slack_alignment term;
       all intermediate terms clipped to [-2,2] for stability and AST depth control."""
    eps = 0.00016500882024926124
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
    ddl_gate = 1.0 / (1.0 + np.exp(-5.910146376480895 * slack))
    slack_alignment = np.clip(norm_slack, -1.0, 1.0)
    ddl_breach = (slack <= 0.0).astype(float)
    critical_path_leverage = norm_rank * norm_work * ddl_breach
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 2.4462574202616563
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-3.9435917035738255 * (slack_pressure - 1.0)))
    boosted_rank = norm_rank * (1.0 + 2.05486654661606 * slack_alignment) * rank_gate
    uncert_gate = 1.0 / (1.0 + np.exp(-5.910146376480895 * (norm_uncert - 0.0009114224441304759)))
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_gate
    wait_benefit = 1.0 - np.exp(-0.837566966667497 * ready_wait_time)
    energy_slack_penalty = norm_energy * (1.0 + 2.4462574202616563 * slack_alignment) * ddl_gate
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(critical_path_leverage, -2.0, 2.0) - np.clip(boosted_rank, -2.0, 2.0) - 1.48966084160097 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + 0.5947490578684853 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 2.4462574202616563 * np.clip(energy_slack_penalty, -2.0, 2.0) + 0.6986001658772105 * np.clip(norm_work, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
