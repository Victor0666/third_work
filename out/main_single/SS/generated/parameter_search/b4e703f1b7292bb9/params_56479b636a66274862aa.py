import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule: merges Parent 2's robust hinge coupling and joint gate with Parent 1's congestion-aware load signal;
       replaces softplus starvation relief with *steepened softplus* to sharpen low-wait sensitivity while preserving saturation;
       introduces *congestion_coupling_strength* to explicitly weight duration-uncertainty co-penalty only under joint feasibility;
       retains median-MAD normalization for outlier resistance and sign stability;
       removes redundant rank_slack_coupling amplification (redundant with slack_pressure_gate) to reduce AST depth;
       enforces strict clipping at [-2,2] on all composite terms to bound gradient magnitude and prevent dominance collapse."""
    eps = 0.0020365276516692054
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
    joint_gate = 1.0 / (1.0 + np.exp(-3.9138159736639064 * slack)) * 1.0 / (1.0 + np.exp(3.9138159736639064 * (norm_uncert - 0.4683916948694513)))
    ddl_breach = (slack <= 0.0).astype(float)
    critical_path_leverage = norm_rank * norm_work * ddl_breach
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.576346598971893
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-1.0950131167400916 * (slack_pressure - 1.0)))
    boosted_rank = norm_rank * (1.0 + 1.4879493856545256 * rank_gate)
    starvation_signal = 0.602160055229832 * (ready_wait_time + eps)
    wait_benefit = np.log1p(np.exp(0.8828339425504893 * starvation_signal))
    congestion_coupling = 0.2508017740477104 * norm_duration * norm_uncert * joint_gate
    energy_uncert_penalty = norm_energy * norm_uncert * joint_gate
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(critical_path_leverage, -2.0, 2.0) - np.clip(boosted_rank, -2.0, 2.0) - 0.2569711811607829 * np.clip(norm_energy * joint_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip(congestion_coupling, -2.0, 2.0) + 0.20637930850980213 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 1.06324691725943 * np.clip(norm_work * joint_gate, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
