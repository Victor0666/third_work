# -*- coding: utf-8 -*-
"""
train_d3qn_single_controller_reward_MIX_Tsize.py  （FCFS + 先空闲先服务 启发式版本）

HRL（阶段式）结构保留，但：
- 去掉所有 D3QN 网络（不再导入 / 创建 / 更新 D3QNAgent）；
- 上层：不再由 Manager 网络 / 模板控制排序，环境内部用 FCFS（先到先服务，同 ready_time 随机）生成阶段任务顺序；
- 下层：不再由 Worker 网络选动作，直接调用环境接口，按“先空闲先服务 + 同时空闲随机”选 VM。

仍然：
- 使用带子 deadline + HEFT-like 的环境 CloudWorkflowEnv_VMAgents；
- 记录每个阶段、每个 episode 的能耗和奖励等信息到 CSV，方便和 HRL-D3QN / P-D3QN 结果对比。
"""

import os
import sys
import numpy as np
import torch  # 仅用于设定随机种子

from project_paths import PROJECT_ROOT
from output_naming import training_output_paths

# 项目根路径
ROOT_DIR = str(PROJECT_ROOT)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from baseline_fcfs.env_fcfs import (
    CloudWorkflowEnv_VMAgents,
)
from common.metrics_logger import CSVLogger


# ---------- 工具 ----------
def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------- 训练主程序（纯启发式） ----------
def train():
    # ---------------- 基本超参 ----------------
    dax_dir = os.path.join(ROOT_DIR, "data", "dax")
    # 启用 5 种工作流模型，环境将按均匀随机抽样到达
    dax_list = [
        os.path.join(dax_dir, "CyberShake_30.xml"),
        os.path.join(dax_dir, "Epigenomics_24.xml"),
        os.path.join(dax_dir, "Ligo_30.xml"),
        os.path.join(dax_dir, "Montage_25.xml"),
        os.path.join(dax_dir, "Sipht_29.xml"),
    ]
    horizon = 1e9
    arrival_lambda = 0.03
    random_seed = 0

    max_ready_tasks = 64
    normalize_obs = True

    # combo_weights 只为兼容构造函数，具体排序逻辑已经在环境中改为 FCFS
    COMBO_WEIGHTS = np.ones(5, dtype=np.float32) / 5.0
    STATE_INCLUDE_VM = True
    STATE_INCLUDE_HOST = True
    STATE_INCLUDE_QUEUE = True

    WORKFLOWS_PER_EPISODE = 50
    MAX_EPISODES = 50  # 你可以按需要调整

    # 训练/评测/存档（这里只是保存日志）
    output_paths = training_output_paths(ROOT_DIR, "fcfs-ss")
    save_dir = str(output_paths.checkpoint_dir)
    log_path = str(output_paths.log_path)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # ---------------- 初始化环境 ----------------
    set_seed(random_seed)

    EnvCls = CloudWorkflowEnv_VMAgents
    env_kwargs = dict(
        dax_paths=dax_list,
        horizon=horizon,
        arrival_lambda=arrival_lambda,
        random_seed=random_seed,
        max_ready_tasks=max_ready_tasks,
        normalize=normalize_obs,
        combo_weights=COMBO_WEIGHTS,
        state_include_vm=STATE_INCLUDE_VM,
        state_include_host=STATE_INCLUDE_HOST,
        state_include_queue=STATE_INCLUDE_QUEUE,
        workflows_per_episode=WORKFLOWS_PER_EPISODE,
        # 资源（25 VM, 3 Hosts：2 Cloud + 1 Edge）
        num_cloud_hosts=2,
        num_edge_hosts=1,
        cloud_vms_per_host=(9, 8),
        edge_vms_per_host=(8,),
        cloud_pc_tiers=(1.0, 2.0, 4.0, 6.0, 8.0),
        edge_pc_tiers=(1.0, 2.0, 4.0, 6.0, 8.0),
        cloud_bw_tiers=(1000.0, 2000.0, 4000.0, 6000.0, 8000.0),
        edge_bw_tiers=(1000.0, 2000.0, 4000.0, 6000.0, 8000.0),
    )

    env = EnvCls(**env_kwargs)
    obs0, info0 = env.reset()

    # 奖励尺度（环境侧）：恢复到温和区间，避免 reward 过大
    # 与 env.__init__ 中默认值保持一致
    env.energy_reward_scale = 1e-3      # 能耗差分 × 1e-3
    env.task_baseline_beta = 0.05       # 任务延迟奖励缩放系数
    env.task_baseline_norm = 300.0      # 同时用于裁剪区间 [-300, 300]（在 env 中实现）

    print(
        f"[scale] energy_reward_scale={env.energy_reward_scale}  "
        f"beta={env.task_baseline_beta}  norm={env.task_baseline_norm}"
    )

    # 把尺度同步回 env_kwargs，方便 reset 后重新写回
    env_kwargs.update({
        "energy_reward_scale": env.energy_reward_scale,
        "task_baseline_beta": env.task_baseline_beta,
        "task_baseline_norm": env.task_baseline_norm,
    })

    # 探测 Worker 观测维度（只是打印一下方便对比）
    st, ok = env.get_worker_state_for_next_assignment()
    if not ok:
        # 若此时还没有 ready + idle 状态，则推进到决策点再试一次
        _ = env.finish_phase_and_advance()
        st, ok = env.get_worker_state_for_next_assignment()
    state_dim_worker = st["obs"].shape[0]
    action_dim_worker = st["mask"].shape[0]
    print(
        f"[env] num_vms={env.num_vms} worker_obs_dim={state_dim_worker} worker_act_dim={action_dim_worker} "
        f"(VM={STATE_INCLUDE_VM}, Host={STATE_INCLUDE_HOST}, Queue={STATE_INCLUDE_QUEUE})"
    )

    # 日志器（字段名沿用原 HRL 版本，便于画图脚本复用）
    logger = CSVLogger(
        filepath=log_path,
        fieldnames=[
            "step", "episode", "type",
            "ep_length", "env_time",
            "episode_energy",
            "wf_completed", "wf_target",
            "epsilon_worker", "epsilon_manager",
            # 阶段指标
            "assign_cnt", "phase_size_sum_mi",
            "r_worker_phase_mean", "r_worker_phase_mean_norm",
            "r_manager_raw", "r_manager_norm",
            "r_manager_train_raw", "r_manager_train_norm", "no_assign_phase",
            # 评测（启发式版本不单独评测，留空）
            "eval_reward_worker", "eval_reward_manager", "eval_energy",
            # 汇总
            "avg_assign_per_step",
        ],
    )

    # ---------------- 纯启发式调度循环 ----------------
    global_step = 0
    episode_idx = 0

    # 本局统计量初始化
    ep_energy0 = float(env.total_energy)
    assign_sum_episode = 0
    episode_steps = 0

    while episode_idx < MAX_EPISODES:
        if env.done_flag:
            # ===== 一局结束：记录本局信息 =====
            ep_energy = float(env.total_energy - ep_energy0)
            avg_assign_per_step = assign_sum_episode / max(episode_steps, 1)

            logger.log(
                step=global_step, episode=episode_idx, type="episode",
                ep_length=episode_steps, env_time=getattr(env, "current_time", 0.0),
                episode_energy=ep_energy,
                wf_completed=getattr(env, "completed_workflows", 0),
                wf_target=(getattr(env, "workflows_per_episode", None) or ""),
                epsilon_worker="",    # 无网络，留空
                epsilon_manager="",   # 无网络，留空
                assign_cnt="", phase_size_sum_mi="",
                r_worker_phase_mean="", r_worker_phase_mean_norm="",
                r_manager_raw="", r_manager_norm="",
                r_manager_train_raw="", r_manager_train_norm="", no_assign_phase="",
                eval_reward_worker="", eval_reward_manager="", eval_energy="",
                avg_assign_per_step=avg_assign_per_step,
            )

            print(
                f"[episode={episode_idx}] len={episode_steps} "
                f"energy={ep_energy:.3f} J | "
                f"avg_assign/step={avg_assign_per_step:.6f} | "
                f"wf={getattr(env,'completed_workflows',0)}/{getattr(env,'workflows_per_episode','∞')}"
            )

            # ===== 开新局 =====
            episode_idx += 1
            if episode_idx >= MAX_EPISODES:
                break

            # 换个随机种子，重新 reset
            env.random_seed = getattr(env, "random_seed", 0) + 1
            obs0, info0 = env.reset()
            # reset 后把奖励尺度写回，防止内部重置
            env.energy_reward_scale = env_kwargs["energy_reward_scale"]
            env.task_baseline_beta = env_kwargs["task_baseline_beta"]
            env.task_baseline_norm = env_kwargs["task_baseline_norm"]

            ep_energy0 = float(env.total_energy)
            assign_sum_episode = 0
            episode_steps = 0

            continue  # 进入下一局循环

        # ===== 阶段内：在当前决策点尽量分配任务 =====
        phase_worker_rewards_raw = []
        phase_assign_cnt = 0

        while True:
            st, has_next = env.get_worker_state_for_next_assignment()
            if not has_next:
                break

            # 下层：采用“先空闲先服务 + 同时空闲随机”选 VM
            # 由环境内部方法 worker_assign_first_idle() 实现
            r_task_raw = float(env.worker_assign_first_idle())
            phase_worker_rewards_raw.append(r_task_raw)
            phase_assign_cnt += 1

        # 阶段末：推进到下一决策点，仅结算能耗 → r_manager_raw（能耗强度）
        r_manager_raw, pinfo = env.finish_phase_and_advance()
        phase_size_sum_mi = float(pinfo.get("phase_size_sum_mi", 0.0))
        assign_cnt = int(pinfo.get("assign_cnt", 0))
        no_assign_phase = 1 if assign_cnt == 0 else 0
        r_worker_phase_mean = float(np.mean(phase_worker_rewards_raw)) if len(phase_worker_rewards_raw) > 0 else 0.0

        logger.log(
            step=global_step, episode=episode_idx, type="phase",
            ep_length="", env_time=pinfo.get("current_time", 0.0),
            episode_energy="", wf_completed="", wf_target="",
            epsilon_worker="", epsilon_manager="",
            assign_cnt=assign_cnt, phase_size_sum_mi=phase_size_sum_mi,
            r_worker_phase_mean=r_worker_phase_mean, r_worker_phase_mean_norm="",
            r_manager_raw=r_manager_raw, r_manager_norm="",
            r_manager_train_raw=r_manager_raw, r_manager_train_norm="",
            no_assign_phase=no_assign_phase,
            eval_reward_worker="", eval_reward_manager="", eval_energy="",
            avg_assign_per_step="",
        )

        # 更新统计
        assign_sum_episode += assign_cnt
        episode_steps += 1
        global_step += 1

    print(
        f"启发式调度模拟结束。\n"
        f"共运行回合数：{episode_idx}\n"
        f"日志保存在：{log_path}"
    )


if __name__ == "__main__":
    train()
