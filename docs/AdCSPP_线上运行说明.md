# AdCSPP 线上运行说明

## 配置边界

- `configs/adcsp_online.yaml`：线上真实数据训练和连续路径生成配置，不生成仿真输入。

线上配置不包含任何仿真数据生成参数；平台 input pipeline 负责读取 TFRecord。

## 入口和 task 链路

```text
run.py
  -> PipelineBuilder
  -> AdsRecPipeline
  -> TaskBuilder
  -> AdCSPPFlowTrainTask / AdCSPPFlowGenerateTask
  -> AdCSPPModel
```

当前新增的两个 task：

- `AdCSPPFlowTrainTask`：读取平台提供的 TFRecord dataset，训练点击 hazard 和点击后转化 hazard，保存 TensorFlow weights checkpoint。
- `AdCSPPFlowPredictTask`：读取平台提供的 TFRecord dataset，调用 `AdCSPPModel.click_probability` 和 `AdCSPPModel.conversion_probability`。
- `AdCSPPFlowGenerateTask`：读取平台提供的真实用户状态作为生成种子，并加载训练阶段从真实 TFRecord 学到的 `AdCSPPFlowPrior`；每一步生成下一次请求上下文，再采样点击，只有点击后才采样转化，然后把曝光、点击、转化结果反馈到下一步状态和历史序列。

这两个 task 不复用原有监督学习 `Train` / `Predict` task。

## 当前闭环能力

| 能力 | 实现位置 |
| --- | --- |
| 学习点击概率 | `AdCSPPFlowTrainTask` + `AdCSPPModel.click_head` |
| 学习转化概率 | `AdCSPPFlowTrainTask` + `AdCSPPModel.conversion_head`，训练时对点击样本做条件化 mask |
| 输入历史序列 | `AdCSPPTFRecordAdapter` + `UserStateBuilder` |
| 生成连续用户路径 | `AdCSPPPathGenerator.rollout` |
| 动态刷新历史序列 | `models/adcsp/model.py::_append_state_history` |
| 曝光—点击—转化状态机 | `AdCSPPPathGenerator`，转化采样强制依赖点击 |
| 输出完整流量轨迹 | `AdCSPPFlowGenerateTask.results["trajectories"]`，每步包含前后状态、上下文、概率和行为结果 |
| 高保真生成 | 由真实 TFRecord 学到的流量 prior、行为 head、可学习 latent transition、历史序列反馈共同完成；上线前应使用离线校准指标和路径级分布指标验收 |

这里的“生成”不涉及重新召回或选择广告。广告请求字段的分布从真实训练流量中学习；模型只在学习到的请求分布上生成下一步请求，并生成用户的点击/转化行为，再把结果反馈给下一步状态。

## 流量模型输出

流量行为模型只输出行为建模结果：

- `pred_click_probability`
- `pred_conversion_probability`

连续生成 task 还输出内存中的轨迹对象，每一步包含：当前请求上下文、点击/转化概率、采样后的 `is_click`、`is_conversion`、生成前状态和生成后状态。它不选择广告，也不负责 JSON 序列化。

credit、trajectory credit、counterfactual uplift 等归因输出不属于 AdCSPP 流量生成模型。

运行示例：

```bash
python run.py --config_path configs/adcsp_online.yaml
```

默认执行顺序是先训练再生成；只运行预测时可以覆盖 step：

```bash
python run.py --config_path configs/adcsp_online.yaml --step adcsp_flow_predict
```

只运行连续路径生成：

```bash
python run.py --config_path configs/adcsp_online.yaml --step adcsp_flow_generate
```

默认情况下，生成 task 使用平台输入的用户状态作为种子，并由训练阶段保存的
`adcsp_flow_prior.json` 生成后续请求。如果需要完全从训练分布采样新用户，可在离线
或回放配置中设置 `generate_from_prior: true`、`num_users` 和 `num_steps`；即使在该模式下，
没有已训练 prior 也会直接报错，不会退化为手写规则流量。

本地没有平台数据时，`inputs/adcsp/synthetic_trajectory.py` 提供仅用于训练和单元测试的逻辑轨迹 fixture。它不属于线上配置，也不替代线上 TFRecord 输入。

初版线上序列输入先收敛为 7 个核心字段：365d 点击 Slot/时间、365d 转化 Slot/时间/事件类型/目标，以及 12h 曝光未点击行业序列。其他兴趣、创意和频次序列暂不进入第一版模型，待核心闭环验证后再逐步接入。

线上标签使用真实字段 `label` 和 `label_keyaction`；离线 fixture 也使用同一组标签和线上特征名。线上 schema 中没有直接用户静态画像，adapter 不会伪造画像字段，而是使用长期统计、实时统计和核心行为序列作为用户状态输入。

离线可以用 `configs/adcsp_offline.yaml` 运行完整闭环：先由逻辑 fixture 形成训练流，训练行为模型和流量 prior，再从 prior 采样新用户并生成完整路径。该配置只用于开发验证。

