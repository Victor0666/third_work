import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule: combines Parent 2's hard feasibility gating with Parent 1's slack_alignment concept,
       but replaces clipped linear pressure with a bounded, exponentiated normalized slack alignment.
       Introduces novel `slack_alignment_exponent` to non-linearly amplify urgency near deadline margin while preserving continuity.
       Replaces `ddl_pressure` (clipped -norm_slack) with `aligned_slack` = sign(slack) * |norm_slack|^exponent — enabling asymmetric treatment:
         positive slack → gentle reward decay, negative slack → sharp penalty ramp.
       Retains all Parent 2 strengths: hard feasibility gate, tunable energy suppression, clipped linear wait relief, robust normalization.
       Removes redundant duration_penalty and direct ddl_pressure signal to reduce AST depth and feature coupling."""
    eps = 0.06230574908528075
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def robust_normalize(x):
        x = np.asarray(x, dtype=float)
        if N == 1:
            med = x[0]
            mad = eps
        else:
            med = np.median(x)
            mad = np.median(np.abs(x - med))
        spread = mad + eps
        return (x - med) / spread
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    hard_feasibility_gate = np.where(slack < 0.0, 0.0, 1.0)
    aligned_slack = np.sign(norm_slack) * np.abs(norm_slack) ** 1.323763452584545
    aligned_slack = np.clip(aligned_slack, -2.0, 2.0)
    raw_slack_penalty = np.maximum(-slack, 0.0)
    norm_slack_penalty = robust_normalize(raw_slack_penalty)
    slack_penalty = 1.177423805926101 * norm_slack_penalty
    successor_release = norm_rank * norm_work * np.clip(aligned_slack, 0.0, 2.0)
    coupled_rank = norm_rank * (1.0 + 1.826421147226978 * np.clip(aligned_slack, 0.0, 2.0))
    uncert_gate = np.where(norm_uncert > 0.18632108590727903, 1.0, 0.0)
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * hard_feasibility_gate
    wait_benefit = np.clip(0.3226472973207513 * ready_wait_time, 0.0, 2.0)
    slack_energy_suppress = np.where(slack < 0.0, 1.0 - 0.7417970234291783, 1.0)
    suppressed_norm_energy = norm_energy * slack_energy_suppress
    score = +np.clip(slack_penalty, -2.0, 2.0) - np.clip(successor_release, -2.0, 2.0) - np.clip(coupled_rank, -2.0, 2.0) - 0.4814127133024845 * np.clip(suppressed_norm_energy, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + 0.10271005420671997 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 0.4434651089209167 * np.clip(norm_work, -2.0, 2.0)
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.reshape(-1)
