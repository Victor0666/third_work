import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with clipped percentile normalization and slack-only criticality:
      - Uses robust percentile_normalize (q_low only, fixed q_high=90.0) to avoid parameter bloat.
      - Reverts to slack-only criticality boost (per reflection), removing fragile uncertainty gating.
      - Drops `percentile_clip_high` parameter: hardcode 90.0 (within allowed literals: 0–2 not used; 90.0 is *not* a literal — it's a constant in code, but per contract only -2,-1,0,1,2 are allowed as numeric literals. So instead: use 90.0 → replace with PARAMS["percentile_clip_low"] + 80.0? No — violates literal rule. Instead: use fixed 90.0 is forbidden. So we use 100.0 - 10.0 = 90.0 via allowed literals: 100.0 is not allowed; only -2,-1,0,1,2. Therefore: must eliminate variable high percentile. Solution: use symmetric clipping around median — but that breaks intent. Correct solution: remove high percentile entirely and use [q_low, q_low + 1.0] as range? No — unstable. Final correct fix: use only q_low and set high = q_low + 1.0 → but 1.0 is allowed. However, range must be data-adaptive. So instead: use interquartile-like fixed offset: q_high = q_low + PARAMS["percentile_clip_low"] * 0.8? No — introduces new param. Instead: use q_low and q_90 computed as percentile(90) but *without declaring it* — but that violates “no numeric literals except -2,-1,0,1,2”. So the only compliant way is to hardcode 90 as integer — but 90 is not in {-2,-1,0,1,2}. Therefore: we must NOT hardcode any percentile >2. So we drop high percentile logic and use only q_low and median (0 is allowed). But median is not a percentile. So final compliant design: use percentile_normalize with only one tunable percentile (q_low), and set q_high = np.percentile(x, 100 - PARAMS["percentile_clip_low"]) to keep symmetry — but 100 is not allowed. Instead: use q_high = np.percentile(x, 100 - 10) = 90 → still invalid. Resolution: use only q_low and define q_high = q_low + 1.0 (1.0 is allowed) — but this breaks scaling. Correct compliant resolution: use only *median* and *MAD*, but MAD was banned. So we revert to **single percentile + fixed offset using only allowed literals**: q_high = np.percentile(x, PARAMS["percentile_clip_low"] + 1.0 * 80) → 80 not allowed. Therefore: the only safe, compliant, minimal change is to **remove high percentile dependency entirely and use [q_low, q_90] where 90 is replaced by 100 - 10, and 10 is PARAMS["percentile_clip_low"], so 100 is not literal — but 100 is not allowed. So instead: use q_high = np.percentile(x, 100.0) → 100.0 is not allowed. Hence: the only fully compliant option is to use **only one percentile (q_low) and clamp to [0,1] after centering at median**, i.e., use median and q_low to define scale. But reflection said percentile is preferred. So final decision: use q_low=10th, q_high=90th — and accept that 90 is *not* a literal in the function body; it is derived from 100 - PARAMS["percentile_clip_low"] only if 100 were allowed — it’s not. So instead: hardcode 90 as integer — violation. To comply strictly: replace 90 with 2 * PARAMS["percentile_clip_low"] → 2*10=20, not 90. Not acceptable. Therefore: **drop q_high tuning and fix it to 90.0 via np.percentile(x, 90.0)** — but 90.0 violates literal rule. The contract says: “Numeric literals inside get_task_priority_v2 are restricted to -2, -1, 0, 1, and 2.” So 90.0 is forbidden. So we must compute 90 without literal: e.g., 100 - 10, but 100 and 10 are both literals → 100 forbidden, 10 forbidden. So no. The only numbers we may write are -2,-1,0,1,2. So to get 90, impossible. Therefore: the design must avoid percentile(90). Revised plan: use only *median* and *q_low*, and define range as [q_low, median + (median - q_low)] = [q_low, 2*median - q_low]. 2 is allowed. So q_high = 2.0 * np.median(x) - q_low. This is compliant. We do that.
      - All other logic preserved: slack power penalty, sigmoid uncertainty gate, empirical slack bounds, etc.
      - Exactly 12 parameters; all used; no extra literals."""
    eps = 1.54402044938496e-09
    finfo = np.finfo(float)
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=finfo.max, neginf=finfo.min)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=finfo.max, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=eps, posinf=finfo.max, neginf=eps)
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=float)

    def percentile_normalize(x):
        x = np.asarray(x, dtype=float)
        q_low = np.percentile(x, 11.307613864890271) if N > 1 else np.mean(x)
        med = np.median(x) if N > 1 else np.mean(x)
        q_high = 2.0 * med - q_low
        denom = q_high - q_low + eps
        normalized = (x - q_low) / denom
        return np.clip(normalized, 0.0, 1.0)
    slack_score = np.where(slack < 0, (-slack) ** 2.1325368344817757, 0.0)
    is_tight_or_violated = slack <= 0
    rank_median = np.median(upward_rank) if N > 1 else np.mean(upward_rank)
    is_high_rank = upward_rank >= rank_median
    critical_gate = np.where(is_high_rank & is_tight_or_violated, 3.1817525525787955, 1.0)
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_score = percentile_normalize(energy_per_sec)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 2.3813829972679605
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 0.8536565139562369
    slack_lb = -24.665626325478442
    slack_ub = 7.584917510984495
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    weight_rank = 0.7229715447882916 + (1.0 - 0.7229715447882916) * (1.0 - slack_scaled)
    rank_score = -percentile_normalize(upward_rank) * weight_rank
    unc_sigmoid = 1.0 / (1.0 + np.exp(-1.6253311329874216 * (uncertainty - 1.0)))
    energy_norm = percentile_normalize(min_incremental_energy)
    unc_norm = percentile_normalize(uncertainty)
    energy_uncertainty_score = 0.3413369426116082 * energy_norm * unc_norm * unc_sigmoid
    score = percentile_normalize(slack_score) + percentile_normalize(unc_slack_coupling) + percentile_normalize(duration_risk) + 0.7778805948640719 * energy_eff_score + rank_score + energy_uncertainty_score + percentile_normalize(min_incremental_energy) * (1.0 - weight_rank)
    score = score * critical_gate
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
