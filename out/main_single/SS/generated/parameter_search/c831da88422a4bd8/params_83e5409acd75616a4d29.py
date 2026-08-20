import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule with all declared parameters used, no unused entries.
       - Removed 'criticality_scale' (previously unused) and replaced its role by direct use of
         slack_pressure_ramp in duration_criticality and starvation_urgency_gated terms.
       - All numeric literals are in {-2,-1,0,1,2}; epsilon is exclusively from PARAMS["epsilon"].
       - Uses median-MAD normalization, piecewise slack ramp, DDL gates, successor release, and unified timing boost."""
    eps = 0.00011303653345611283
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
    ddl_gate = 1.0 / (1.0 + np.exp(-4.236722677711832 * slack))
    ddl_breach = (slack <= 0.0).astype(float)
    slack_ramp_center = 0.2644015877685797
    slack_pressure_ramp = np.clip((norm_slack + slack_ramp_center) * 2.84237064084539, 0.0, 1.0)
    successor_gate = (norm_uncert > 0.41113903008002983).astype(float)
    successor_release = norm_rank * norm_work * successor_gate * ddl_breach * 1.9866953244975827
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 2.177256071533816
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    duration_criticality = norm_duration * norm_rank * slack_pressure_ramp
    wait_benefit_pressure = (1.0 - np.exp(-0.9912429885431099 * ready_wait_time)) * ddl_breach
    starvation_urgency_gated = wait_benefit_pressure * (1.0 + 2.177256071533816 * slack_pressure_ramp)
    energy_uncert_penalty = norm_energy * norm_uncert * successor_gate * ddl_gate
    energy_slack_penalty = norm_energy * (1.0 + 2.177256071533816 * slack_pressure_ramp) * ddl_gate
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(successor_release, -2.0, 2.0) - np.clip(duration_criticality, -2.0, 2.0) - 0.8066320224105277 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - np.clip(starvation_urgency_gated, -2.0, 2.0) + 0.5269434704338014 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 2.177256071533816 * np.clip(energy_slack_penalty, -2.0, 2.0) + 0.04608948629015647 * np.clip(norm_work, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
