# AdCSPP 与设计文档逐模块对齐检查

本文对应 `docs/AdCSPP_设计文档_V1.2.md`，用于检查当前代码是否已经形成“从真实训练日志学习，再生成连续用户路径”的闭环。

## 1. 结论

V1 条件行为生成链路已经打通：

```text
真实 TFRecord
  -> UserState / 历史序列
  -> StaticUserEncoder + ContextEncoder + SequenceEncoder
  -> Continuous-time State Backbone
  -> Click / Conversion Hazard
  -> HierarchicalEventGenerator
  -> FiniteStateMachine + EventLedger
  -> DynamicStateTransition + DynamicStateUpdater
  -> 下一次曝光上下文和下一次用户状态
  -> generated_traffic.jsonl
```

这里的生成对象是完整的用户曝光路径；曝光上下文、静态画像和历史序列来自训练数据学习到的 prior，点击和转化由行为模型生成，状态机和更新器保证路径合法并刷新下一步输入。

## 2. 六个核心模块逐项对齐

| 设计模块 | 当前实现 | 训练/生成输入 | 输出与验证 | 状态 |
| --- | --- | --- | --- | --- |
| Static User Encoder | `models/adcsp/encoders.py::StaticUserEncoder` | 用户静态画像、长期统计特征 | 固定维度 user embedding | 已实现 |
| Dynamic State Transition | `models/adcsp/state_transition.py::DynamicStateTransition`，由 `AdCSPPModel.differentiable_transition()` 调用 | 上一时刻 latent、静态 embedding、上下文 embedding、序列 embedding、click/conversion mark | 下一时刻 latent state | 已实现 |
| Exposure Opportunity Generator | `models/adcsp/opportunity.py::ExposureOpportunityGenerator` | `AdCSPPFlowPrior` 学到的用户路径长度、曝光间隔、上下文转移 | 下一次曝光时间和上下文 | V1 已实现 |
| Hierarchical Click/Conversion Generator | `models/adcsp/event_generator.py::HierarchicalEventGenerator` | 当前用户状态、当前曝光上下文 | 先 click，再基于 click 条件生成 conversion；同时生成窗口内事件时间 | 已实现 |
| Finite State Machine | `models/adcsp/fsm.py::FiniteStateMachine` + `EventLedger` | 曝光、点击、转化事件 | 约束顺序、时间窗口、事件关联并记录事件账本 | 已实现 |
| Dynamic Feature Updater | `models/adcsp/state_updater.py::DynamicStateUpdater` | FSM 确认后的事件 | 更新计数器、最近事件字段和核心序列特征 | 已实现 |

## 3. 各模块的实际接口

### 3.1 Static User Encoder

代码入口：`StaticUserEncoder.call(profile)`。

- 类别字段使用稳定 hash + embedding，避免线上依赖固定词表。
- 数值字段直接进入投影层。
- 用户状态中的 `static_features` 与 `dynamic_features` 不会被混入新的仿真字段。
- 输入缺失时使用 `unknown/0`，保证线上 TFRecord 的字段缺失不会中断推理。

### 3.2 Dynamic State Transition

代码入口：`DynamicStateTransition.call(...)`。

它只负责潜在状态转移，不负责修改显式序列，避免同一事件被重复计数。`AdCSPPModel.transition_state()` 统一调用该模块，再调用 `DynamicStateUpdater`。

### 3.3 Exposure Opportunity Generator

代码入口：`ExposureOpportunityGenerator.next(profile, timestamp, previous_context)`。

V1 的设计边界是“曝光由平台输入”，因此训练和条件生成仍可直接使用平台曝光上下文。完整路径生成任务需要从训练流量生成下一次曝光时，使用 `AdCSPPFlowPrior` 中真实数据估计的：

- 用户轨迹长度分布；
- 用户曝光间隔分布；
- 上一次 slot 到下一次上下文的经验转移；
- 用户静态画像、长期统计和历史序列。

这不是固定规则生成器；没有拟合 prior 时，`generate()` 会直接报错，防止无训练数据时静默生成假流量。

### 3.4 Hierarchical Click/Conversion Generator

代码入口：`HierarchicalEventGenerator.generate(...)`。

生成顺序固定为：

1. 根据 `P(click | state, exposure)` 采样 click；
2. 将 click mark 作为条件，计算 `P(conversion | state, exposure, click)`；
3. 只有 click=1 时才允许 conversion=1；
4. 在 click window / conversion window 内采样事件发生时间。

当前转换目标继承曝光上下文的 `promotion_target` 或 `crtvid_conversion_target`，并由 `DynamicStateUpdater` 写入对应序列。

### 3.5 Finite State Machine

代码入口：`FiniteStateMachine.register_exposure()`、`record_click()`、`record_no_click()`、`record_conversion()`。

约束包括：

- click 必须关联已登记 exposure；
- click 不能早于 exposure，且不能超过 click window；
- conversion 必须关联 click；
- conversion 不能早于 click，且不能超过 conversion window；
- 每次事件都进入 `EventLedger`，生成输出保留事件记录。

### 3.6 Dynamic Feature Updater

代码入口：`DynamicStateUpdater.apply_outcome()`。

它是显式动态状态的唯一写入口：

- 曝光：更新 exposure count、最近曝光 slot/time；
- 点击：追加 `auid_hwdsp_clk_slotid_seq_time_365d` 和 click timestamp；
- 未点击：追加 `rt_auid_hwdsp_imp_unclk_3rd_indid_seq_time_12h`；
- 转化：追加转化 slot、timestamp、event type、conversion target，并更新计数器。

所有序列都按 `max_sequence_length` 截断，且不会覆盖初始历史快照 `initial_sequence_features`。

## 4. 输出是否满足流量生成目标

生成任务输出的每一行包含：

- 用户和曝光上下文：`auid/user_id`、`slot_id`、上下文特征、`logid`/时间字段；
- 行为标签：`is_click`、`is_conversion`；
- 模型概率：`click_probability`、`conversion_probability`；
- 事件账本：`events`，含 exposure/click/conversion/no-click 及其时间；
- 更新后的核心序列特征和动态状态字段。

因此可以按用户重新排序，得到长度不固定的连续曝光路径。`num_records` 只控制最终输出总行数；未配置固定步数时，每个用户的路径长度从真实训练轨迹长度分布采样。

## 5. 当前明确的 V1 边界

以下能力属于设计文档中的 V2 或 S2P2 完整实现，当前没有冒充已经完成：

1. 当前 Continuous-time State Backbone 是可训练的连续时间衰减骨干，不是完整论文级 S2P2 / NHP 点过程实现。
2. 当前 Exposure Opportunity Generator 使用训练数据经验 prior；尚未训练独立的 `lambda_opp(t)` 曝光强度 head。
3. 当前没有广告候选选择、预算/频控/资格约束模块；平台已给定曝光时不需要这些模块。
4. 当前点击/转化概率是窗口级 hazard 概率，事件时间使用与窗口概率一致的截断指数采样；如果后续需要目标级多风险过程，需要增加 target-specific heads。
5. 当前 FSM 已支持多个 impression 的账本结构，但生成器按串行曝光上下文 rollout；完整的优先队列并发事件调度属于后续增强。

这些边界不影响 V1 的核心验证目标：静态特征保持、动态序列随事件刷新、曝光—点击—转化链路合法、并从真实训练数据学习生成流量。

## 7. V2 已新增能力

V2 代码已继续补齐以下模块：

| V2 能力 | 代码入口 | 说明 |
| --- | --- | --- |
| Exposure Opportunity Head | `models/adcsp/opportunity.py::ExposureOpportunityHead` | 输出正的曝光机会强度；支持指数间隔 NLL 训练 |
| Ad Selection Head | `models/adcsp/heads.py::AdSelectionHead` | 对候选广告上下文打分并做 masked softmax |
| Candidate Masking | `models/adcsp/heads.py::CandidateMasker` | 支持 slot、屏蔽广告和频控约束 |
| Joint Event Scheduler | `models/adcsp/scheduler.py::JointEventScheduler` | 统一按时间处理曝光机会、点击、转化和未点击终止 |
| Joint Traffic Generator | `models/adcsp/joint_generator.py::JointTrafficGenerator` | 自主曝光机会 + 候选选择 + 行为生成闭环 |
| V2 Joint Mode | `tasks/adcsp_flow.py::AdCSPPFlowGenerateTask` | 通过 `generation_mode: joint` 复用原有 generate task |
| Joint Loss | `losses/adcsp_losses.py` | 曝光机会强度 NLL 和候选选择交叉熵 |
| Candidate Selection Trainer | `models/adcsp/heads.py::CandidateSelectionTrainer` | 使用候选集合和实际选中 index 训练选择 head |

V2 的候选池必须由平台上游传入，代码不会写死候选广告。相关参数统一放在 `configs/adcsp_online.yaml` 的 `adcsp_flow_generate` 节点，并将 `generation_mode` 改为 `joint`。

当前机会强度训练已支持从真实用户的曝光间隔训练 `ExposureOpportunityHead`，通过 `adcsp_flow_train.opportunity_training.enabled: true` 开启。候选选择 head 可通过 `CandidateSelectionTrainer` 使用“候选集合 + 实际选中候选 index/ID”训练；没有这类字段时，只能进行候选推理和约束校验，不能声称已经完成选择分布训练。

## 6. 单元测试覆盖

`D:\Anaconda\python.exe -m unittest discover -s tests` 当前通过 16 项测试，覆盖：

- 编码器与 hazard 概率有限且在 `[0,1]`；
- 模型状态转移输出合法；
- FSM 的曝光—点击—转化顺序与非法转化拒绝；
- DynamicStateUpdater 的计数与核心序列刷新；
- 路径 rollout 的状态前后快照；
- learned flow prior 的轨迹长度和上下文采样；
- 生成任务的 TFRecord 训练、预测与流量文件输出。

