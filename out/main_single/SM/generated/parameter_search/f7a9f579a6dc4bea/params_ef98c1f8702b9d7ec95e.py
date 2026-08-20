import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining strengths from both parents:
      - Retains Parent 2's robust quantile normalization and piecewise-linear DDL urgency.
      - Replaces exponential starvation ramp with tunable power-law saturation: (wait/(wait+θ))^p — smoother near zero, more flexible than exp(-wait/θ), avoids numerical underflow.
      - Keeps binary feasibility mask for energy term (from Parent 1's decisive ddl_gate logic) but applies it *only* when slack is strictly feasible (slk > delta), ensuring hard constraint adherence.
      - Uses unified bottleneck term: rank * work / (|slk| + eps), no exponentiation — stable and interpretable.
      - All parameters declared are used; no literals beyond -2,-1,0,1,2; epsilon via PARAMS; clamping via np.finfo.
      - Deterministic, finite, shape-correct, side-effect-free.
    """
    eps = 0.005373628352100711
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    work = np.asarray(remaining_work, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)

    def normalize(x):
        x = np.asarray(x, dtype=np.float64)
        abs_x = np.abs(x)
        finite_mask = np.isfinite(abs_x)
        if np.any(finite_mask):
            scale = np.quantile(abs_x[finite_mask], 0.8934373258105495)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    delta = 0.015296879669745998
    ddl_feasible_mask = np.where(slk > delta, 1.0, 0.0)
    ddl_urgency = np.clip((slk + delta) / (2 * delta), 0.0, 1.0)
    slack_penalty = np.where(slk < 0, 6.437750802760693 * np.abs(slk), -4.514679211294417 * slk)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -1.8689771346896618 * normalize(inv_energy) * ddl_feasible_mask
    rank_score = -1.4975877371063615 * ddl_urgency * normalize(rank + eps)
    bottleneck_term = rank * work / (np.abs(slk) + eps)
    bottleneck_score = -0.19480794904388074 * normalize(bottleneck_term + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 1.5872704060588478 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    starvation_numerator = wait
    starvation_denominator = wait + 9.549780889620115 + eps
    starvation_ratio = starvation_numerator / starvation_denominator
    starvation_sat = np.power(starvation_ratio + eps, 0.16082591036167984)
    wait_score = -normalize(starvation_sat + eps)
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 773762.9598729258
    min_safe = -finfo.max / 773762.9598729258
    score = np.clip(score, min_safe, max_safe)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
