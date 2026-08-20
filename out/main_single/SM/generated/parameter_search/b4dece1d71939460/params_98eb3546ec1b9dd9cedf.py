import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule with smooth DDL protection gate and successor-release interaction.
    
    Key structural mutations:
      - Replaces hard `slack_norm >= 0` gating with *smooth sigmoid gate* centered at slack=0,
        controlled by `ddl_protection_gate_width` — eliminates discontinuities while preserving
        strong deadline-driven activation near zero slack.
      - Adds `load_successor_release_weight`-scaled term: computes normalized product of
        `upward_rank` and `remaining_work`, then applies smooth gate — promotes tasks that
        release large downstream critical work early under deadline pressure.
      - Retains robust normalization, exponential wait saturation, and bounded linear duration-uncertainty blend.
      - All gates are monotonic and differentiable; no conditionals on raw magnitudes.
      - Uses `np.tanh` for numerically stable, bounded smooth step instead of fragile piecewise logic.
    """
    eps = 0.002216156569838458
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    rem_work = np.asarray(remaining_work, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)

    def normalize(x):
        x = np.asarray(x)
        abs_x = np.abs(x)
        scale = np.mean(abs_x) if np.all(np.isfinite(abs_x)) and np.mean(abs_x) > eps else eps
        return x / (scale + eps)
    gate_center = 0.0
    gate_width = 0.05294960476610147 + eps
    slack_gate = np.tanh((slk - gate_center) / gate_width)
    slack_penalty = 2.5161745913478266 * np.where(slk < 0, np.abs(slk), 0.0) - 4.583292949869763 * np.where(slk > 0, slk, 0.0)
    slack_penalty = normalize(slack_penalty + eps)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.23764555686250594 * normalize(inv_energy)
    rank_score = -2.433772169486276 * normalize(rank + eps) * (1.0 + slack_gate) / 2.0
    successor_release_potential = rank * rem_work
    successor_release_score = -1.1396847801519996 * normalize(successor_release_potential + eps) * (1.0 - slack_gate) / 2.0
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.9258221826160259 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_sat = 1.0 - np.exp(-0.295472499430518 * wait)
    wait_score = -normalize(wait_sat + eps)
    unc_slack_interaction = 3.044976126129415 * uncert_norm * np.maximum(0.0, -slack_gate)
    score = slack_penalty + energy_score + rank_score + successor_release_score + dur_score + wait_score + unc_slack_interaction
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 183191.77889291587
    min_safe = -finfo.max / 183191.77889291587
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
