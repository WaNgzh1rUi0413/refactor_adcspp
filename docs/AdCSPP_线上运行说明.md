# AdCSPP 线上运行说明

## 配置边界

- `configs/adcsp_online.yaml`：线上真实数据预测配置，不生成仿真数据。
- `configs/adcsp_synthetic.yaml`：本地无数据时的仿真配置，仅用于单元测试和行为链路验证。

线上配置不包含 `num_users`、`exposures_per_user`、`horizon_seconds`、随机种子或仿真窗口等数据生成参数。

## 入口和 task 链路

```text
run.py
  -> PipelineBuilder
  -> AdsRecPipeline
  -> TaskBuilder
  -> AdCSPPFlowPredictTask
  -> AdCSPPModel
```

当前新增的两个 task：

- `AdCSPPFlowGenerateTask`：本地仿真流量生成，内部调用 `AdCSPPModel`。
- `AdCSPPFlowPredictTask`：线上 JSONL 条件行为预测，内部调用 `AdCSPPModel.click_probability` 和 `AdCSPPModel.conversion_probability`。

这两个 task 不复用原有监督学习 `Train` / `Predict` task。

## 线上预测输出

`AdCSPPFlowPredictTask` 保留原始线上字段，并追加：

- `pred_click_probability`
- `pred_conversion_probability`
- `expected_future_conversion`

运行示例：

```bash
python run.py --config_path configs/adcsp_online.yaml
```

