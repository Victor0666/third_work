import numpy as np

# 函数名中的 2 由 SeEvo 在生成 Prompt 时替换为目标版本号。
# 八个输入均为长度 N 的一维数组；相同下标始终指向同一个 ready task。
# 返回值也必须是长度 N 的一维数组，并遵守“分数越小，优先级越高”。
def get_task_priority_v2(
    min_exec_time,
    min_comm_time,
    min_incremental_energy,
    slack,
    upward_rank,
    remaining_work,
    ready_wait_time,
    uncertainty
):

    """
    v2 evolution: Fixes multiplicative distortion by reverting to *additive, hierarchically weighted composition*;
    restores full rank dynamic range (no [-1,1] clamping) for stable gradient-sensitive prioritization;
    extends slack-conditioning to *all risk-aware terms* (not just fairness/uncertainty) — e.g., 
    uncertainty penalty now activates *proportionally* under both slack deficit and surplus to prevent deadline fragility;
    replaces CPDR with *normalized critical-path pressure* (CPP = upward_rank / (min_exec_time + min_comm_time + eps)) 
    scaled by global slack distribution to emphasize bottlenecks only when deadlines are tight;
    redefines SEER as *slack-gated latency-efficiency ratio* with smooth exponential decay only below zero slack,
    preserving energy optimization in safe regions without artificial suppression;
    introduces *wait-aware urgency modulation*: boosts urgency term for tasks that have waited long *relative to their slack*, 
    preventing starvation even under positive slack; all operations strictly protected and deterministic.
    """
    eps = 1e-08
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    
    def normalize_rank(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size == 1:
            return np.array([0.0])
        sorted_x = np.sort(x)
        ranks = np.searchsorted(sorted_x, x, side='left')
        ranks_norm = ranks / max(len(sorted_x) - 1, 1)
        return 2.0 * ranks_norm - 1.0  # full [-1,1] range retained for gradient sensitivity
    
    # Adaptive urgency: steepness increases with median slack deficit; avoids hard thresholds
    slack_median = np.median(slack)
    tau_urg = np.clip(1.0 + 0.6 * np.maximum(0.0, -slack_median), 0.2, 6.0)
    urgency_raw = 1.0 / (1.0 + np.exp((slack + 0.5) / tau_urg))
    norm_urgency = normalize_rank(urgency_raw)
    urgency_term = -6.0 * norm_urgency  # dominant weight; smaller score = higher priority
    
    # Wait-aware urgency modulation: penalize long wait *relative to slack margin*, not absolute
    relative_wait = np.where(slack > eps, ready_wait_time / (slack + eps), ready_wait_time + 1e3)
    norm_rel_wait = normalize_rank(relative_wait)
    wait_modulation = 0.4 * np.clip(norm_rel_wait, 0.0, 1.0)  # additive boost to urgency term for starved tasks
    urgency_term = urgency_term + wait_modulation
    
    # Critical-path pressure (CPP): upward_rank normalized by latency cost; scaled by global slack pressure
    latency_cost = min_exec_time + min_comm_time + eps
    cpp_base = upward_rank / latency_cost
    norm_cpp = normalize_rank(cpp_base)
    # Amplify CPP only when slack is tight (median slack < 0), else attenuate to avoid over-prioritizing non-critical paths
    slack_pressure_factor = np.clip(1.0 + 0.8 * np.maximum(0.0, -slack_median), 0.2, 1.8)
    cpp_term = 1.5 * norm_cpp * slack_pressure_factor
    
    # SEER: latency-to-energy ratio, gated *only* under violation (slack < 0) via smooth exponential
    seer_base = latency_cost / (min_incremental_energy + eps)
    seer_gate = np.exp(-np.maximum(0.0, -slack) / 2.0)  # soft gate: full value when slack >= 0, decays smoothly
    seer_gated = seer_base * seer_gate
    norm_seer = normalize_rank(seer_gated)
    seer_term = 1.2 * norm_seer  # positive weight: higher efficiency → lower score (since norm_seer ∈ [-1,1])
    
    # Uncertainty penalty: applied *continuously* across slack spectrum — high uncertainty always harms predictability
    # But scaled by communication cost and normalized rank to avoid dominance
    unc_penalty_base = min_comm_time * uncertainty * 0.1
    # Apply globally, but modulate intensity based on whether slack is extreme (very negative or very positive)
    slack_extremity = np.abs(slack) / (np.maximum(np.std(slack), eps) + eps)
    unc_penalty_scaled = unc_penalty_base * (0.5 + 0.5 * np.tanh(slack_extremity))
    norm_unc = normalize_rank(unc_penalty_scaled)
    unc_term = 0.25 * norm_unc
    
    # Fairness term: active for *all* tasks, but strongest when slack < 0; prevents starvation without breaking hierarchy
    fairness_base = np.sqrt(np.maximum(ready_wait_time, 0.0) + eps) / (np.abs(slack) + 1.0)
    norm_fairness = normalize_rank(fairness_base)
    fairness_term = 0.3 * np.clip(norm_fairness, 0.0, 1.0)
    
    # Final additive composition preserves ordinal integrity and hierarchical intent
    score = (
        urgency_term +
        cpp_term +
        seer_term +
        fairness_term +
        unc_term
    )
    
    # Ensure finite output, no NaN/inf leakage
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    return score
