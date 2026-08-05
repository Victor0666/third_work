# 对比算法目录

后续对比算法统一放在：

```text
algorithms/comparisons/<method_id>/
```

每个方法应使用稳定、唯一的 `method_id`，并至少说明：

- 算法入口和配置；
- 使用的 observation、reward/cost 和动作信息；
- 是否使用 LLM、shield、Q_c、fallback 或示范预训练；
- 与当前方法共享的数据、seed、DDL 和模糊参数；
- 禁止访问的额外信息；
- 对应测试和实验 manifest。

当前已有 `fcfs/`，它是从根目录 `baseline_fcfs/` 迁入的历史 FCFS 对比实现。
根目录同名包仅为旧导入兼容层，不再存放算法实现。

当前新增 `drlea_nichgp/`：它把压缩包2的 DRL-EA/Niching-GP 三阶段机制
适配到本项目统一的动态模糊云边工作流环境。该方法复用问题模拟器和评价协议，
但不调用主算法的三层 Agent、LLM、SeEvo 或安全强化学习组件；入口、状态、
奖励、公平性和上游映射见其 [README](drlea_nichgp/README.md)。
