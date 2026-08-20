import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule: combines Parent 2's robust min-max normalization and raw slack fidelity
       with Parent 1's conditional energy-relaxation under DDL stress and explicit energy-uncertainty penalty.
       Introduces novel dual-gate structure: (1) slack-pressure-triggered energy relaxation,
       and (2) joint energy-uncertainty penalty only active under high risk AND negative slack.
       Uses bounded min-max scaling for all features except slack (kept raw for deadline signal integrity).
       All operations protected against NaN/inf; deterministic and shape-correct."""
    eps = 0.0006493280167470755
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def bounded_minmax(x):
        x_min = np.min(x)
        x_max = np.max(x)
        range_val = x_max - x_min + eps
        return (x - x_min) / range_val
    norm_energy = bounded_minmax(min_incremental_energy)
    norm_duration = bounded_minmax(min_exec_time + min_comm_time)
    norm_rank = bounded_minmax(upward_rank)
    norm_wait = bounded_minmax(ready_wait_time)
    norm_uncert = bounded_minmax(uncertainty)
    slack_pressure_raw = np.clip(-slack, 0.0, None)
    slack_pressure = np.power(slack_pressure_raw + eps, 1.2539688475126696)
    norm_slack_for_gate = bounded_minmax(slack)
    rank_gate = np.clip(1.0 - 2.0 * np.clip(norm_slack_for_gate, 0.0, 0.26248157379662407), 0.0, 1.0)
    boosted_rank = norm_rank * (1.0 + 0.8060172765720927 * rank_gate)
    joint_risk_gate = ((norm_uncert > 0.17946402241735637) & (slack_pressure > 0)).astype(float)
    duration_risk_interaction = norm_duration * joint_risk_gate * 0.32100550170170306
    wait_benefit = np.tanh(0.6313345985705726 * norm_wait)
    coupled_rank = np.clip(norm_rank * (1.0 + 0.2898821101883847 * slack_pressure), 0.0, 2.0)
    energy_weight = np.where(slack <= 0.0, 0.709646831200434 * 0.0107263614646182, 0.709646831200434)
    energy_uncert_active = (slack < 0.0) & (norm_uncert > 0.17946402241735637)
    energy_uncert_penalty = np.where(energy_uncert_active, norm_energy * norm_uncert, np.zeros_like(norm_energy))
    score = +slack_pressure + energy_weight * norm_energy - coupled_rank - wait_benefit + duration_risk_interaction + 0.6944818780321226 * energy_uncert_penalty
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
