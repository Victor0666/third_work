"""SeEvo 启发式规则进化框架的 Hydra 主入口。

对 cews_task_constructive 而言，本入口只负责组合配置、初始化 LLM 客户端并启动
SeEvo；实际云边工作流评价在问题目录的 eval.py 子进程中完成。
"""

from __future__ import annotations
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, ListConfig

from seevo import SeEvo
from utils.utils import init_client


def _case_numbers(value) -> list[int]:
    """把 Hydra 可能产生的空值、列表、逗号字符串或标量统一成整数 seed 列表。"""
    if value is None:
        return []
    if isinstance(value, (list, tuple, ListConfig)):
        return [int(item) for item in value]
    if isinstance(value, str):
        return [int(item) for item in value.replace(",", " ").split() if item]
    return [int(value)]


@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main(cfg: DictConfig) -> None:
    """组合问题配置并运行完整 SeEvo 进化流程。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    if str(cfg.algorithm).lower() != "seevo":
        raise ValueError(
            f"This project snapshot provides the SeEvo implementation only; got {cfg.algorithm!r}."
        )

    # LLM 客户端只服务于规则生成/反思/交叉/变异；评价器本身不调用 LLM。
    init_client(cfg)
    llm_root = Path(__file__).resolve().parent
    # 将 case_num 标准化后交给 SeEvo；空列表表示使用问题 YAML 中的 train/test seeds。
    algorithm = SeEvo(cfg, str(llm_root), _case_numbers(cfg.case_num))
    best_code, best_code_path = algorithm.evolve()
    logging.info("Best candidate path: %s", best_code_path)
    logging.info("Best candidate code:\n%s", best_code)


if __name__ == "__main__":
    main()
