# 当前训练特征整理与分类

## 1. 文档说明

本文档基于当前 `AdsRecPipeline` 的训练配置整理，训练模型为 `ESMMModel`。

当前输入配置包含：

| 类型 | 数量 | 说明 |
|---|---:|---|
| Sparse Sequence | 32 | 用户行为、内容兴趣、点击/转化历史等序列特征 |
| Sparse | 59 | 广告、广告位、App/Site、用户长期统计等离散/数值特征 |
| Dense | 2 | 两个实时统计特征 |
| Label | 2 | CTR / CVR 相关标签，不属于输入特征 |
| Extra Input | 2 | `auid`、`slot_id`，主要用于样本/仿真过程，不直接作为模型 Feature Op 输入 |
| **模型输入特征合计** | **93** | 32 + 59 + 2 |

> **重要说明**
>
> 当前配置中没有直接的用户静态画像字段，例如 `gender`、`age_group`、`region`、`device_type`、`os_type` 等。
> 当前用户侧信息主要通过**长期行为统计 + 短期行为序列 + 点击/转化历史**隐式表达。
>
> 因此，如果后续用于 AdCSPP 流量生成，建议将特征划分为：
>
> 1. User Static Profile
> 2. User Long-term State
> 3. User Short-term / Sequential State
> 4. Current Context
> 5. Real-time Context

---

# 2. 总体 Feature Schema

```text
                    User
                     │
        ┌────────────┴────────────┐
        │                         │
 User Static Profile       User Historical State
  当前基本没有                  │
                         ┌──────┴──────┐
                         │             │
                   Long-term       Sequential
                     State           State
                         │             │
                         └──────┬──────┘
                                │
                                ▼
                       Current Context
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
             Slot              Ad/Creative       App/Site
              │                 │                 │
              └─────────────────┼─────────────────┘
                                │
                                ▼
                         ESMM / CTR-CVR
```

---

# 3. User Static Profile

## 3.1 当前状态

当前训练配置中**没有直接输入的用户静态画像特征**。

例如以下字段目前均不存在：

```text
gender
age
age_group
region
device_type
os_type
user_segment
user_long_term_interest
user_registration_age
user_value_segment
```

因此当前模型不是：

```text
User Static Profile + Context → CTR/CVR
```

而更接近：

```text
User Historical Behavior + Current Context → CTR/CVR
```

## 3.2 对 AdCSPP 的建议

如果后续目标是生成完整用户流量，建议单独建立：

| 特征 | 含义 | 当前是否存在 |
|---|---|---|
| `gender` | 用户性别 | 否 |
| `age_group` | 用户年龄段 | 否 |
| `region` | 用户地域 | 否 |
| `device_type` | 设备类型 | 否 |
| `os_type` | 操作系统 | 否 |
| `user_segment` | 用户分群 | 否 |
| `user_long_term_interest` | 用户长期兴趣 | 否 |
| `user_registration_age` | 用户注册时长 | 否 |
| `user_value_segment` | 用户价值分层 | 否 |

这些属于后续可以补充的**真正 User Static Profile**。

---

# 4. User Long-term State

这一部分描述用户过去较长时间窗口内的行为统计。

## 4.1 用户长期行为统计

| Feature | 含义 | 时间窗口 |
|---|---|---|
| `auid_site_click_cnt_90d` | 用户在 Site 上的点击次数 | 90d |
| `auid_normalizedt2_click_cnt_180d` | 用户对 normalized tag2 类别的点击次数 | 180d |
| `auid_ctr_90d` | 用户整体历史 CTR | 90d |
| `auid_show_cnt_180d` | 用户总曝光次数 | 180d |
| `auid_app_show_cnt_180d` | 用户 App 曝光次数 | 180d |
| `auid_app_click_cnt_90d` | 用户 App 点击次数 | 90d |
| `auid_app_ctr_90d` | 用户 App CTR | 90d |
| `auid_appc3_ctr_180d` | 用户在 App 三级分类上的 CTR | 180d |
| `auid_slot_show_cnt_90d` | 用户广告位曝光次数 | 90d |
| `auid_slotid_imp_cnt_30d` | 用户广告位曝光次数 | 30d |
| `auid_slot_ctr_90d` | 用户广告位 CTR | 90d |
| `auid_slot_click_cnt_90d` | 用户广告位点击次数 | 90d |
| `auid_site_show_cnt_90d` | 用户 Site 曝光次数 | 90d |
| `auid_site_ctr_90d` | 用户 Site CTR | 90d |
| `auid_normalizedt2_ctr_90d` | 用户在 normalized tag2 类别上的 CTR | 90d |

这些特征可以理解为用户的**长期行为状态 / implicit user profile**。

---

# 5. User Short-term / Sequential State

这是当前特征体系中非常重要的一部分，主要描述用户近期行为轨迹。

## 5.1 用户近期点击行为

| Feature | 含义 | 时间窗口 | 最大长度 |
|---|---|---:|---:|
| `rt_auid_alldsp_clk_slotform_seq_cnt_24h` | 用户近期点击的广告位形式及相关次数序列 | 24h | 20 |
| `rt_auid_click_advertiser_list_7d` | 用户近期点击过的广告主列表 | 7d | 20 |
| `rt_auid_hwdsp_clk_prmt_pkgname_seq_time_24h` | 用户点击过的推广 App/Package Name 及时间序列 | 24h | 20 |
| `rt_auid_hwdsp_clk_advrid_seq_time_24h` | 用户点击过的广告主 ID 及时间序列 | 24h | 20 |
| `rt_auid_hwdsp_clk_appc3_seq_cnt_24h` | 用户点击对应的 App 三级分类及次数 | 24h | 20 |
| `rt_auid_alldsp_clk_site_agappc3_seq_time_24h` | 用户点击 Site/App 三级分类相关序列及时间 | 24h | 20 |
| `rt_auid_hwdsp_clk_3rd_indid_seq_time_24h` | 用户点击过的三级行业 ID 及时间 | 24h | 20 |
| `rt_auid_click_appc3_list_3d` | 用户点击过的 App 三级分类列表 | 3d | 20 |
| `rt_auid_hwdsp_clk_1st_indid_seq_time_24h` | 用户点击过的一级行业 ID 及时间 | 24h | 20 |
| `rt_auid_hwdsp_clk_crtv_clstidh_seq_time_24h` | 用户点击过的创意 Cluster ID 及时间 | 24h | 20 |
| `rt_auid_hwdsp_clk_appc3_seq_time_24h` | 用户点击过的 App 三级分类及时间 | 24h | 20 |

---

## 5.2 用户曝光但未点击

| Feature | 含义 | 时间窗口 |
|---|---|---:|
| `rt_auid_hwdsp_imp_unclk_3rd_indid_seq_time_12h` | 用户近期曝光但没有点击的三级行业序列及时间 | 12h |

该特征具有明显的**负反馈**含义：

```text
Exposure
   ↓
No Click
   ↓
用户对该类内容没有产生点击
```

对于后续生成模型，可以把它理解为用户状态更新中的一种 rejection / negative feedback。

---

# 6. 用户创意 / 内容兴趣历史

| Feature | 含义 |
|---|---|
| `auid_hwdsp_reeng_unified_crtv_clstid_time_ubd_list` | 用户再营销/再触达相关的统一创意 Cluster ID 及时间信息 |
| `auid_hwdsp_clk_unified_crtv_clstid_time_ubd_list` | 用户点击过的统一创意 Cluster ID 及时间信息 |
| `auid_hwdsp_imp_unified_crtv_clstid_time_top20_30d` | 用户近 30d 曝光相关的 Top20 创意 Cluster 序列 |
| `auid_siteid_hwdsp_clk_app_id_dts_ubd_list` | 用户点击涉及的 Site/App ID 及时间相关序列 |

> `UBD` 等内部缩写的具体业务定义仅凭当前训练配置无法完全确定，因此本文不对其内部含义做进一步推断。

---

# 7. Click / Conversion Long Sequence

这一组是当前训练特征中对 AdCSPP 最重要的序列特征之一。

## 7.1 Click Sequence

| Feature | 含义 | 时间窗口 |
|---|---|---:|
| `auid_hwdsp_clk_slotid_seq_time_365d` | 用户点击过的广告位 ID 序列 | 365d |
| `auid_hwdsp_clk_ts_seq_time_365d` | 用户点击行为对应的时间戳序列 | 365d |

二者结合可以形成：

```text
Slot_A → Slot_B → Slot_C → ...
  ↓        ↓        ↓
 t1       t2       t3
```

用于描述用户长期点击路径。

---

## 7.2 Conversion Sequence

| Feature | 含义 | 时间窗口 |
|---|---|---:|
| `auid_hwdsp_conv_slotid_seq_time_365d` | 用户发生转化时对应的广告位 ID 序列 | 365d |
| `auid_hwdsp_conv_ts_seq_time_365d` | 用户转化发生时间戳序列 | 365d |
| `auid_hwdsp_conv_evttyp_seq_time_365d` | 用户发生的转化事件类型序列 | 365d |
| `auid_hwdsp_conv_ocpx_trgt_seq_time_365d` | 用户转化对应的目标序列 | 365d |

这部分能够表达：

```text
Click
  ↓
Conversion
  ↓
Conversion Event Type
  ↓
Conversion Target
  ↓
Conversion Slot
  ↓
Conversion Time
```

因此对后续**曝光 → 点击 → 转化的路径生成**尤其重要。

---

# 8. User Real-time State

| Feature | 含义 | 时间窗口 |
|---|---|---:|
| `rt_auid_appid_alldsp_imp_cnt_10minute` | 用户对 App 的全 DSP 曝光次数 | 10min |
| `gender_appid_click_cnt_today` | 当前用户所属 gender 对应群体今天在相关 App 上的点击统计 | Today |
| `gender_slotid_ctr_today` | 当前用户所属 gender 对应群体今天在相关 Slot 上的 CTR | Today |

特别需要注意：

```text
gender_appid_click_cnt_today
gender_slotid_ctr_today
```

不是用户原始 `gender` 字段，而是**基于 gender 条件聚合出来的实时统计特征**。

---

# 9. Current Context — Creative

当前广告/素材本身的属性。

| Feature | 含义 |
|---|---|
| `creative_image_size` | 创意图片尺寸 |
| `is_video` | 是否为视频创意 |
| `is_short_video` | 是否为短视频 |
| `creativetype` | 创意类型 |
| `creative_ocr_text_keywords` | 创意图片 OCR 文本关键词 |
| `creative_text_keywords` | 创意文本关键词 |
| `title_text_keywords` | 广告标题文本关键词 |
| `video_prmt_label_class3` | 视频推广内容的三级标签 |
| `crtv_clstid_stream` | 创意 Cluster ID 相关序列 |
| `app_introduction` | App/广告介绍文本相关特征 |

---

# 10. Current Context — App / Site

| Feature | 含义 |
|---|---|
| `app_class1` | App 一级分类 |
| `app_class2` | App 二级分类 |
| `app_class3` | App 三级分类 |
| `app_type` | App 类型 |
| `game_type` | 游戏类型 |
| `site_app_allias_id` | Site/App 别名 ID |
| `site_unique_id` | Site 唯一 ID |
| `media_type_new` | 媒体类型 |
| `site_click_cnt_30d` | Site 近 30d 点击次数 |
| `site_paid_cnt_30d` | Site 近 30d 付费次数 |
| `site_paid_rate_smooth_3d` | Site 近 3d 平滑付费率 |
| `site_activate_cnt_14d` | Site 近 14d 激活次数 |
| `site_browse_cnt_14d` | Site 近 14d 浏览次数 |
| `site_click_cnt_14d` | Site 近 14d 点击次数 |
| `site_register_count_14d` | Site 近 14d 注册次数 |
| `site_retain_cnt_14d` | Site 近 14d 留存次数 |
| `site_show_cnt_14d` | Site 近 14d 曝光次数 |
| `site_register_hour_top_30d` | Site 近 30d 注册高频时间段 |
| `site_paid_hour_top3_30d` | Site 近 30d 付费高频时间段 |

---

# 11. Current Context — Advertiser / Task

| Feature | 含义 |
|---|---|
| `indu_name` | 广告/业务所属行业 |
| `task_name` | 当前广告任务名称/任务类型 |
| `developer_up_id` | 开发者/广告主 ID |
| `developer_up_realname` | 开发者/广告主名称 |
| `objective_type` | 广告优化/目标类型 |
| `promotion_target` | 推广目标 |
| `crtvid_conversion_target` | 当前创意对应的转化目标 |

---

# 12. Current Context — Slot

| Feature | 含义 |
|---|---|
| `slot_form` | 广告位形式 |
| `slot_suited_creative_type` | 广告位适配的创意类型 |
| `slot_quality_score` | 广告位质量评分 |
| `slot_star_ranking_cn` | 广告位星级/排名相关指标 |
| `slot_avg_click_bid_ecpm_3d` | Slot 近 3d 点击相关平均出价/eCPM |
| `slot_ctr_3rdpty_dsp_7d` | Slot 近 7d 第三方 DSP CTR |
| `slot_ctr_3rdpty_dsp_30d` | Slot 近 30d 第三方 DSP CTR |
| `ecpm_floor_price` | 广告位 eCPM 底价 |
| `price_strategy` | 当前价格策略 |

---

# 13. Current Context — Category / Tag

| Feature | 含义 |
|---|---|
| `unifiled_tag_level1_all_ssp` | SSP 统一一级标签 |
| `unifiled_tag_level2_all_ssp` | SSP 统一二级标签 |
| `unifiled_tag_level3` | 统一三级标签 |
| `normalized_tag2_id` | normalized tag2 ID |
| `slotid_click_unify_prmt_class1_freq_rank_list` | Slot 点击相关一级推广分类频次/排名列表 |
| `slotid_click_unify_prmt_class3_freq_rank_list` | Slot 点击相关三级推广分类频次/排名列表 |

---

# 14. Current Context — Real-time Ad Statistics

| Feature | 含义 |
|---|---|
| `rt_adid_hwdsp_pred_loan_clk_credit_score_bin_30minute` | 广告近 30min 预测贷款点击信用分分桶 |
| `rt_adid_hwdsp_pred_loan_clk_credit_score_30minute` | 广告近 30min 预测贷款点击信用分 |

---

# 15. 其他 Context 特征

| Feature | 含义 |
|---|---|
| `unifiled_tag_level1_all_ssp` | SSP 统一一级标签 |
| `unifiled_tag_level2_all_ssp` | SSP 统一二级标签 |
| `normalized_tag2_id` | 标准化二级标签 ID |
| `slotid_click_unify_prmt_class1_freq_rank_list` | Slot 历史点击对应一级推广类别排名 |
| `slotid_click_unify_prmt_class3_freq_rank_list` | Slot 历史点击对应三级推广类别排名 |

---

# 16. Labels：点击与转化监督信号

当前训练配置包含两个行为标签。它们不是普通输入特征，而是模型需要学习和预测的**监督目标**；AdCSPP 也必须保留这两个标签对应的行为结果。

| 原始 Label | AdCSPP 统一名称 | 含义 | 监督的行为 |
|---|---|---|---|
| `label` | `click_label` | 当前曝光样本是否发生点击，通常为 0/1 | Exposure → Click / No Click |
| `label_keyaction` | `conversion_label` | 当前曝光样本是否发生 Key Action 转化，通常为 0/1 | Exposure / Click → Conversion / No Conversion |

当前 ESMM 配置的字段映射为：

```text
ctr_label = label
conversion_label = label_keyaction
ctcvr_label = ctcvr
```

其预测关系可概括为：

```text
label             → CTR / Click Probability
label_keyaction   → CVR / Key Action Conversion Probability
CTR × CVR         → CTCVR（联合点击转化概率）
```

## 16.1 AdCSPP 必须建模的两个结果变量

对每一条曝光样本，AdCSPP 都需要学习或生成两个结果：

```text
Exposure_t
   │
   ├── click_label = 0  → No Click
   │
   └── click_label = 1  → Click
                              │
                              ├── conversion_label = 0 → No Conversion
                              │
                              └── conversion_label = 1 → Key Action Conversion
```

因此，AdCSPP 不能只生成正向点击和转化事件，还必须生成并保留：

- 曝光后未点击的样本（`click_label = 0`）；
- 点击后未转化的样本（`click_label = 1, conversion_label = 0`）；
- 发生点击并最终转化的样本（两个标签均为 1）。

这些负向结果是用户行为分布的重要组成部分，不能在生成过程中丢弃。

## 16.2 标签与事件序列的关系

`label` 和 `label_keyaction` 是针对当前训练样本的结果标签；它们本身通常只表达是否发生行为，不直接提供行为发生时间、广告位或转化目标。

这些事件属性需要结合历史序列及原始事件日志获得：

| 信息 | 对应字段 / 来源 | 在 AdCSPP 中的用途 |
|---|---|---|
| 点击是否发生 | `label` | 决定是否生成 Click 事件 |
| 点击发生时间 | 点击事件时间戳 / `auid_hwdsp_clk_ts_seq_time_365d` | 生成 Click 时间，并更新用户状态 |
| 点击发生的广告位 | `auid_hwdsp_clk_slotid_seq_time_365d` | 记录点击对应的 Slot |
| 转化是否发生 | `label_keyaction` | 决定是否生成 Key Action Conversion 结果 |
| 转化发生时间 | 转化事件时间戳 / `auid_hwdsp_conv_ts_seq_time_365d` | 生成 Conversion 时间，并更新用户状态 |
| 转化对应广告位 | `auid_hwdsp_conv_slotid_seq_time_365d` | 记录转化对应的 Slot |
| 转化事件类型 | `auid_hwdsp_conv_evttyp_seq_time_365d` | 生成具体转化事件类型 |
| 转化目标 | `auid_hwdsp_conv_ocpx_trgt_seq_time_365d` | 生成具体 Conversion Target |

**重要：** 标签的归因口径、转化窗口以及 `label_keyaction` 是否只包含点击归因转化，需要以现有 Label 生产逻辑为准。若它是点击归因标签，则生成时应约束转化发生在对应点击之后，并遵守归因窗口；若标签还包含曝光归因/其他归因类型，则需要按真实归因规则保留相应路径，不能仅凭字段名称强制规定“没有点击就绝不可能转化”。

## 16.3 防止标签泄漏

在预测当前曝光的点击和转化结果时，不能把当前样本的 `label` 或 `label_keyaction` 当作已知输入特征，否则会发生标签泄漏。

正确方式是：

1. 使用曝光前的 User State 和当前 Context 预测/采样 `click_label`；
2. 根据点击结果及真实归因规则预测/采样 `conversion_label`；
3. 将生成的 Click / Conversion 事件及其时间、类型、目标写入事件历史；
4. 更新为下一时刻的 User State，供后续曝光使用。

历史样本中已经发生的点击/转化可以作为历史状态输入；当前样本尚未发生的标签只能作为监督目标或待生成结果。

---

# 17. Extra Input

以下两个字段存在于输入配置中，但不作为普通 Feature Op 特征：

| Feature | 用途 |
|---|---|
| `auid` | 用户 ID，用于用户级数据组织、仿真等 |
| `slot_id` | 广告位 ID，用于当前 Slot 标识和仿真 |

当前 Simulation 配置也明确使用：

```text
user_id_field = auid
slot_id_field = slot_id
```

---

# 18. 面向 AdCSPP 的最终 Feature Schema

如果后续把当前 ESMM 特征体系迁移到 AdCSPP，我建议重新组织成下面的结构。

## 18.1 User Static Profile

```text
gender
age_group
region
device_type
os_type
user_segment
user_long_term_interest
user_registration_age
user_value_segment
```

**当前：基本为空，需要后续补充。**

---

## 18.2 User Long-term State

```text
auid_site_click_cnt_90d
auid_normalizedt2_click_cnt_180d
auid_ctr_90d
auid_show_cnt_180d
auid_app_show_cnt_180d
auid_app_click_cnt_90d
auid_app_ctr_90d
auid_appc3_ctr_180d
auid_slot_show_cnt_90d
auid_slotid_imp_cnt_30d
auid_slot_ctr_90d
auid_slot_click_cnt_90d
auid_site_show_cnt_90d
auid_site_ctr_90d
auid_normalizedt2_ctr_90d
```

---

## 18.3 User Short-term State

```text
rt_auid_appid_alldsp_imp_cnt_10minute

rt_auid_alldsp_clk_slotform_seq_cnt_24h
rt_auid_click_advertiser_list_7d
rt_auid_hwdsp_clk_prmt_pkgname_seq_time_24h
rt_auid_hwdsp_clk_advrid_seq_time_24h
rt_auid_hwdsp_clk_appc3_seq_cnt_24h
rt_auid_alldsp_clk_site_agappc3_seq_time_24h
rt_auid_hwdsp_clk_3rd_indid_seq_time_24h
rt_auid_click_appc3_list_3d
rt_auid_hwdsp_clk_1st_indid_seq_time_24h
rt_auid_hwdsp_clk_crtv_clstidh_seq_time_24h
rt_auid_hwdsp_clk_appc3_seq_time_24h

rt_auid_hwdsp_imp_unclk_3rd_indid_seq_time_12h
```

---

## 18.4 User Click / Conversion State

```text
auid_hwdsp_clk_slotid_seq_time_365d
auid_hwdsp_clk_ts_seq_time_365d

auid_hwdsp_conv_slotid_seq_time_365d
auid_hwdsp_conv_ts_seq_time_365d
auid_hwdsp_conv_evttyp_seq_time_365d
auid_hwdsp_conv_ocpx_trgt_seq_time_365d
```

这一层建议在 AdCSPP 中单独处理，因为它天然包含：

```text
Event Type
Timestamp
Slot
Conversion Target
```

非常适合构建用户行为状态转移。

---

## 18.5 Current Slot Context

```text
slot_id
slot_form
slot_suited_creative_type
slot_quality_score
slot_star_ranking_cn
slot_avg_click_bid_ecpm_3d
slot_ctr_3rdpty_dsp_7d
slot_ctr_3rdpty_dsp_30d
ecpm_floor_price
price_strategy
```

---

## 18.6 Current Ad / Advertiser Context

```text
indu_name
task_name
developer_up_id
developer_up_realname
objective_type
promotion_target
crtvid_conversion_target
```

---

## 18.7 Current Creative Context

```text
creative_image_size
creativetype
is_video
is_short_video

creative_ocr_text_keywords
creative_text_keywords
title_text_keywords
video_prmt_label_class3
crtv_clstid_stream
app_introduction
```

---

## 18.8 Current App / Site Context

```text
app_class1
app_class2
app_class3
app_type
game_type

site_app_allias_id
site_unique_id
media_type_new

site_click_cnt_30d
site_paid_cnt_30d
site_paid_rate_smooth_3d

site_activate_cnt_14d
site_browse_cnt_14d
site_click_cnt_14d
site_register_count_14d
site_retain_cnt_14d
site_show_cnt_14d

site_register_hour_top_30d
site_paid_hour_top3_30d
```

---

## 18.9 Click / Conversion Labels 与 AdCSPP 行为生成

AdCSPP 除了接收 User State 和 Current Context，还需要显式输出当前曝光对应的点击、转化结果。建议在模型接口中统一使用以下名称，并保留与原始训练标签的映射：

| AdCSPP 字段 | 原始训练字段 | 类型 | 说明 |
|---|---|---|---|
| `click_label` | `label` | Binary（0/1） | 当前曝光是否产生点击 |
| `conversion_label` | `label_keyaction` | Binary（0/1） | 当前曝光样本是否产生 Key Action 转化 |

### 18.9.1 单次曝光的输入与输出

**输入：**

```text
UserStatic
UserLongTermState
UserShortTermState
HistoricalExposureClickConversionEvents
CurrentContext
```

**输出：**

```text
ExposureEvent:
  user_id
  exposure_timestamp
  slot_id
  current_context

ClickOutcome:
  click_label: 0 | 1
  click_timestamp: optional

ConversionOutcome:
  conversion_label: 0 | 1
  conversion_timestamp: optional
  conversion_event_type: optional
  conversion_target: optional
```

当 `click_label = 0` 时，不生成当前曝光对应的 Click 事件；当 `conversion_label = 0` 时，不生成当前样本对应的 Key Action Conversion 事件。时间、事件类型和目标等字段只在相应事件真实发生时生成，并且必须满足时间与归因规则。

### 18.9.2 行为生成与状态更新

```text
UserState_t + CurrentContext_t
              │
              ▼
         Exposure_t
              │
              ▼
      Predict / Sample click_label
              │
       ┌──────┴──────┐
       │             │
  click_label=0  click_label=1
       │             │
    No Click      Click Event
                     │
                     ▼
          Predict / Sample conversion_label
                     │
              ┌──────┴──────┐
              │             │
     conversion_label=0  conversion_label=1
              │             │
         No Conversion  Conversion Event
                            │
                            ▼
                 Update User State
                            │
                            ▼
                       UserState_{t+1}
```

上图表达的是点击归因转化场景下的默认路径。若当前标签口径包含曝光归因转化或其他转化路径，应根据真实归因类型扩展状态转移分支。

### 18.9.3 训练目标建议

AdCSPP 至少需要保留两个可分别评估的监督目标：

- **点击目标：** 根据曝光前用户状态和当前上下文，学习 `P(click_label = 1 | UserState_t, Context_t, Exposure_t)`。
- **转化目标：** 根据曝光上下文、点击结果、用户历史状态及归因规则，学习 `P(conversion_label = 1 | UserState_t, Context_t, ClickOutcome_t)`。

如果转化标签只对点击归因样本有效，转化分支应按该标签口径进行条件建模；如果标签包含其他归因路径，则应显式输入/建模归因类型。不要在未核实标签定义前，把 `label_keyaction` 简化成无条件的“点击后转化概率”。

### 18.9.4 与历史序列特征的配合

`click_label` / `conversion_label` 负责表达当前样本的行为结果；365d 点击和转化序列负责提供已经发生的历史事件。二者共同形成训练和生成闭环：

```text
历史 Click / Conversion Sequences
              +
     User State_t + Context_t
              │
              ▼
      click_label / conversion_label
              │
              ▼
      生成带时间戳的事件记录
              │
              ▼
       更新历史事件序列与 User State
              │
              ▼
            下一次曝光
```

注意：生成当前结果时，不能把当前样本的两个标签提前放进输入；只有在结果生成后，才能将其作为后续时刻的历史状态。

---

# 19. 对 AdCSPP 最重要的特征关系

最终可以把当前特征理解成：

```text
                     User Static
                         │
                         ▼
              ┌────────────────────┐
              │ Long-term User     │
              │ State              │
              └─────────┬──────────┘
                        │
                        ▼
              ┌────────────────────┐
              │ Short-term User   │
              │ State              │
              └─────────┬──────────┘
                        │
                        ▼
              ┌────────────────────┐
              │ Historical Events  │
              │                    │
              │ Exposure           │
              │   ↓                │
              │ Click              │
              │   ↓                │
              │ Conversion         │
              └─────────┬──────────┘
                        │
                        ▼
             ┌──────────────────────┐
             │ Current Context      │
             │                      │
             │ Slot                 │
             │ Ad                   │
             │ Creative             │
             │ App / Site           │
             │ Industry             │
             │ Price                │
             └──────────┬───────────┘
                        │
                        ▼
                 Behavior Model
                        │
              ┌─────────┼─────────┐
              ▼         ▼         ▼
           Exposure    Click   Conversion
                        ▲         ▲
                        │         │
                    label     label_keyaction
                 (click_label) (conversion_label)
                        │         │
                        └────┬────┘
                             ▼
                        State Update
                        │
                        ▼
                    State(t+1)
```

---

# 20. 对当前特征体系的关键判断

### 20.1 当前最大的缺口：User Static Profile

当前没有真正的：

```text
用户年龄
用户性别
用户地域
设备
OS
长期兴趣标签
用户价值分层
```

所以如果 AdCSPP 需要生成“具有稳定用户身份”的长期流量，建议补充这一层。

### 20.2 当前最有价值的是 User Sequential State

尤其是：

```text
auid_hwdsp_clk_slotid_seq_time_365d
auid_hwdsp_clk_ts_seq_time_365d

auid_hwdsp_conv_slotid_seq_time_365d
auid_hwdsp_conv_ts_seq_time_365d
auid_hwdsp_conv_evttyp_seq_time_365d
auid_hwdsp_conv_ocpx_trgt_seq_time_365d
```

它们可以提供：

```text
行为发生了什么
+
在哪里发生
+
什么时候发生
+
最终是否转化
+
转化目标是什么
```

### 20.3 当前 ESMM 没有显式建模 State Transition

目前的 ESMM 本质仍然是：

```text
User Historical Features
        +
Current Context
        ↓
     ESMM
        ↓
 CTR / CVR / CTCVR
```

而 AdCSPP 更适合：

```text
State_t
   +
Context_t
   ↓
Exposure_t
   ↓
Click_t?
   ↓
Conversion_t?
   ↓
State_{t+1}
```

因此，后续如果以这些训练特征为基础开发 AdCSPP，重点不应该只是“把 93 个特征换一个生成模型”，而应该进一步定义：

**哪些特征是固定状态、哪些特征随曝光更新、哪些特征由 Click 更新、哪些特征由 Conversion 更新，以及不同事件发生后 State 如何转移。**

---

## 21. 当前特征体系一句话总结

> **当前 ESMM 的特征体系本质上是“用户历史行为状态 + 当前广告/广告位上下文”，其中用户静态画像缺失，而点击/转化时间序列已经具备较强的行为轨迹信息。**

这套划分可以直接作为后续 **AdCSPP Feature Schema / Input-Output Design** 的基础版本。
