# AdCSPP 线上运行说明

## 配置边界

- `configs/adcsp_online.yaml`：线上真实数据训练和预测配置，不生成仿真数据。

线上配置不包含任何仿真数据生成参数；平台 input pipeline 负责读取 TFRecord。

## 入口和 task 链路

```text
run.py
  -> PipelineBuilder
  -> AdsRecPipeline
  -> TaskBuilder
  -> AdCSPPFlowTrainTask / AdCSPPFlowPredictTask
  -> AdCSPPModel
```

当前新增的两个 task：

- `AdCSPPFlowTrainTask`：读取平台提供的 TFRecord dataset，训练点击 hazard 和点击后转化 hazard，保存 TensorFlow weights checkpoint。
- `AdCSPPFlowPredictTask`：读取平台提供的 TFRecord dataset，调用 `AdCSPPModel.click_probability` 和 `AdCSPPModel.conversion_probability`。

这两个 task 不复用原有监督学习 `Train` / `Predict` task。

## 流量模型输出

流量行为模型只输出行为建模结果：

- `pred_click_probability`
- `pred_conversion_probability`

credit、trajectory credit、counterfactual uplift 等归因输出不属于 AdCSPP 流量生成模型。

运行示例：

```bash
python run.py --config_path configs/adcsp_online.yaml
```

默认执行顺序是先训练再预测；只运行预测时可以覆盖 step：

```bash
python run.py --config_path configs/adcsp_online.yaml --step adcsp_flow_predict
```

