import functools

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    precision_recall_curve,
    auc as pr_auc_score
)

import utils.config.constants as Const
from utils.math import weird_division


def convert_tensors_to_numpy_and_back(func):
    """
    将 tf.Tensor 输入转为 NumPy 数组，执行计算后再转回 tf.Tensor
    用于计算训练或验证时的 metric
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        has_tensor_input = False

        new_args = list(args)
        # 假设前两个参数是 y_true, y_pred
        for i, arg in enumerate(args[:2]):
            if isinstance(arg, tf.Tensor):
                has_tensor_input = True
                new_args[i] = arg.numpy()

        new_kwargs = kwargs.copy()
        for key in ['y_true', 'y_pred']:
            if key in kwargs and isinstance(kwargs[key], tf.Tensor):
                has_tensor_input = True
                new_kwargs[key] = kwargs[key].numpy()

        result = func(*new_args, **new_kwargs)

        # 如果输入包含tensor，将结果转回tensor
        if has_tensor_input:
            if isinstance(result, tuple):
                result = tuple(tf.convert_to_tensor(r) for r in result)
            else:
                result = tf.convert_to_tensor(result)
        return result

    return wrapper


def convert_dataframe_to_numpy(func):
    """
    将 dataframe 输入转为 numpy 向量，
    用于Evaluate时处理dataframe数据
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        new_kwargs = kwargs.copy()
        y_true_key = kwargs.get('y_true_key', 'y_true')
        y_pred_key = kwargs.get('y_pred_key', 'y_pred')
        if 'eval_df' in kwargs:
            if args:
                raise ValueError("`eval_df` cannot be used with positional arguments")
            new_kwargs['y_true'] = kwargs['eval_df'][y_true_key].values
            new_kwargs['y_pred'] = kwargs['eval_df'][y_pred_key].values
            if 'group_col' in kwargs:
                if isinstance(kwargs['eval_df'][kwargs['group_col']].iloc[0], bytes):
                    kwargs['eval_df'][kwargs['group_col']] = kwargs['eval_df'][kwargs['group_col']].apply(
                        lambda x: x.decode('utf-8'))
                if kwargs['group_col'] == 'column_id' and 'column_index' in kwargs:
                    new_kwargs['group_col'] = kwargs['eval_df']['column_id'].str.split('#').str.get(
                        int(kwargs['column_index'])).to_numpy()
                else:
                    new_kwargs['group_col'] = kwargs['eval_df'][kwargs['group_col']].values
        else:
            if len(args) >= 2:
                new_kwargs['y_true'] = args[0]
                new_kwargs['y_pred'] = args[1]
                args = args[2:]
            else:
                raise ValueError("At least 'y_true' and 'y_pred' positional arguments are required")
            if len(args) >= 1:
                new_kwargs['group_col'] = args[0]
                args = args[1:]
        result = func(*args, **new_kwargs)
        return result

    return wrapper


def clip_predictions(func):
    @functools.wraps(func)
    def wrapper(y_true, y_pred, **kwargs):
        # 将y_pred限制在[Const.EPS, 1-Const.EPS]范围内
        clipped_y_pred = np.clip(y_pred, Const.EPS, 1 - Const.EPS)
        return func(y_true, clipped_y_pred, **kwargs)

    return wrapper


@convert_dataframe_to_numpy
@convert_tensors_to_numpy_and_back
def auc(y_true, y_pred, **kwargs):
    return roc_auc_score(y_true, y_pred)


@convert_dataframe_to_numpy
@convert_tensors_to_numpy_and_back
@clip_predictions
def logloss(y_true, y_pred, **kwargs):
    return log_loss(y_true, y_pred)


@convert_dataframe_to_numpy
@convert_tensors_to_numpy_and_back
@clip_predictions
def bucket_copc(y_true, y_pred, buckets=10, **kwargs):
    """分桶COPC"""
    eval_df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    eval_df['bucket'] = pd.qcut(eval_df['y_pred'], buckets, duplicates='drop')
    grouped = eval_df.groupby('bucket', observed=False).agg(
        pctr=('y_pred', 'mean'),
        ctr=('y_true', 'mean')
    )
    grouped = grouped[grouped.pctr > 0]
    return (grouped.ctr / grouped.pctr).mean()


@convert_dataframe_to_numpy
@convert_tensors_to_numpy_and_back
@clip_predictions
def bucket_pcoc(y_true, y_pred, buckets=10, **kwargs):
    """分桶PCOC"""
    eval_df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    eval_df['bucket'] = pd.qcut(eval_df['y_pred'], buckets, duplicates='drop')
    grouped = eval_df.groupby('bucket', observed=False).agg(
        pctr=('y_pred', 'mean'),
        ctr=('y_true', 'mean')
    )
    grouped = grouped[grouped.ctr > 0]
    return (grouped.pctr / grouped.ctr).mean()


@convert_dataframe_to_numpy
@convert_tensors_to_numpy_and_back
@clip_predictions
def pcoc(y_true, y_pred, **kwargs):
    """预测点击率与实际点击率比值"""
    pctr = y_pred.mean()
    ctr = y_true.mean()
    return pctr / ctr if ctr != 0 else 0.0


@convert_dataframe_to_numpy
@convert_tensors_to_numpy_and_back
def count(y_true, y_pred, **kwargs):
    return len(y_pred)


@convert_dataframe_to_numpy
@convert_tensors_to_numpy_and_back
def mean(y_true, y_pred, **kwargs):
    return y_pred.mean()


@convert_dataframe_to_numpy
@convert_tensors_to_numpy_and_back
def variance(y_true, y_pred, **kwargs):
    return y_pred.var()


@convert_dataframe_to_numpy
@convert_tensors_to_numpy_and_back
def stddev(y_true, y_pred, **kwargs):
    return y_pred.std()


def _segments_ratio(y_true, y_pred, segments, precision, num_col, den_col):
    """
    segments: '0#0.1#...#1.0'
    num_col: 分子列名 'sum_pred' 或 'sum_true'
    den_col: 分母列名 'sum_true' 或 'sum_pred'
    return: ','.join(格式化后的桶比率)
    """
    try:
        bins = list(map(float, segments.split('#')))
        if len(bins) < 2:
            raise ValueError("segments must contain at least two values")
    except (ValueError, AttributeError) as e:
        raise ValueError("segments must be a string of numbers separated by '#'") from e

    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    df['bucket'] = pd.cut(df['y_pred'], bins=bins, include_lowest=True)

    agg = df.groupby('bucket', observed=False).agg(
        sum_true=('y_true', 'sum'),
        sum_pred=('y_pred', 'sum'),
        count=('y_true', 'count')
    )

    agg['ratio'] = np.divide(
        agg[num_col] + Const.EPS,
        agg[den_col] + Const.EPS,
    )
    agg.loc[agg['count'] == 0, 'ratio'] = 0.0

    fmt = f'{{:.{precision}f}}'
    return ','.join(fmt.format(r) for r in agg['ratio'])


@convert_dataframe_to_numpy
@convert_tensors_to_numpy_and_back
@clip_predictions
def segments_copc(y_true, y_pred, segments='0.0#0.1#0.2#0.3#0.4#0.5#0.6#0.7#0.8#0.9#1.0', precision=3, **kwargs):
    """指定分桶点 COPC"""
    return _segments_ratio(y_true, y_pred, segments, precision, num_col='sum_true', den_col='sum_pred')


@convert_dataframe_to_numpy
@convert_tensors_to_numpy_and_back
@clip_predictions
def segments_pcoc(y_true, y_pred, segments='0.0#0.1#0.2#0.3#0.4#0.5#0.6#0.7#0.8#0.9#1.0', precision=3, **kwargs):
    """指定分桶点 PCOC"""
    return _segments_ratio(y_true, y_pred, segments, precision, num_col='sum_pred', den_col='sum_true')


@convert_dataframe_to_numpy
@convert_tensors_to_numpy_and_back
def pr_auc(y_true, y_pred, **kwargs):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    return pr_auc_score(recall, precision)


def gen_group_samples(y_true, y_pred, group_col):
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'group_col': group_col})
    return df.groupby('group_col', observed=False)


@convert_dataframe_to_numpy
@clip_predictions
def gauc(y_true, y_pred, group_col, **kwargs):
    """分组AUC"""
    df = pd.DataFrame({'label': y_true, 'score': y_pred, 'group_id': group_col})
    sorted_data = df.sort_values(['group_id', 'score'], ascending=[True, False])
    sorted_data['group_cum_pos'] = sorted_data.groupby('group_id')['label'].cumsum()
    sorted_data["neg_group_cum_sum"] = (1 - sorted_data["label"]) * sorted_data["group_cum_pos"]
    group_stats = sorted_data.groupby('group_id').agg(
        pos_sum=('label', 'sum'),
        total_samples=('label', 'count'),
        neg_cum_pos=('neg_group_cum_sum', 'sum')
    ).reset_index(drop=True)

    # 过滤只有正样本和只有负样本的情况
    group_stats['neg_sum'] = group_stats['total_samples'] - group_stats['pos_sum']
    group_stats = group_stats[
        (group_stats['pos_sum'] > 0) & (group_stats['neg_sum'] > 0)]

    # 计算各分组的AUC
    group_stats['auc'] = group_stats['neg_cum_pos'] / (group_stats['pos_sum'] * group_stats['neg_sum'])
    # 计算GAUC
    result = weird_division((group_stats['auc'] * group_stats['total_samples']).sum(),
                            group_stats['total_samples'].sum())
    return result


@convert_dataframe_to_numpy
@clip_predictions
def gpcoc(y_true, y_pred, group_col, **kwargs):
    """分组PCOC"""
    grouped = gen_group_samples(y_true, y_pred, group_col)

    grouped = grouped.agg(
        pctr=('y_pred', 'mean'),
        ctr=('y_true', 'mean'),
        cnt=('y_true', 'count')
    )
    # grouped.ctr为0的数据代表无正样本，在此基础上无法计算pcoc，因此需要过滤
    grouped = grouped[grouped.ctr > 0]

    total_pcoc = 0.0
    total_weight = 0
    for _, group in grouped.iterrows():
        pctr = group['pctr']
        ctr = group['ctr']
        cnt = group['cnt']
        pcoc_value = weird_division(pctr, ctr)
        total_pcoc += pcoc_value * cnt
        total_weight += cnt
    return weird_division(total_pcoc, total_weight)


def calculate_error(group):
    sum_gt = group['y_true'].sum()
    sum_scores = group['y_pred'].sum()
    if sum_gt == 0 or sum_scores == 0:
        return np.nan
    ratio = weird_division(sum_gt, sum_scores) if sum_gt > sum_scores else weird_division(sum_scores, sum_gt)
    return (ratio - 1) ** 2


@convert_dataframe_to_numpy
@clip_predictions
def cal_n(y_true, y_pred, group_col, **kwargs):
    y_pred[y_pred <= Const.EPS] = 0.0
    grouped = gen_group_samples(y_true, y_pred, group_col)
    return np.sqrt(grouped.apply(calculate_error).mean(skipna=True))


@convert_dataframe_to_numpy
@clip_predictions
def abs_bias(y_true, y_pred, group_col, **kwargs):
    """
    计算整体绝对值 bias 指标
    """
    grouped = gen_group_samples(y_true, y_pred, group_col)

    grouped = grouped.agg(
        count=('y_true', 'count'),
        sum_y=('y_true', 'sum'),
        avg_y=('y_true', 'mean'),
        avg_p=('y_pred', 'mean')
    ).reset_index()

    global_mean = grouped['avg_y'].mean()
    global_var = grouped['avg_y'].var(ddof=0)  # 总体方差
    # 防止除 0
    if global_var == 0:
        overall_alpha = overall_beta = 1.0
    else:
        tmp = weird_division(global_mean * (1 - global_mean), global_var) - 1
        overall_alpha = global_mean * tmp
        overall_beta = tmp * (1 - global_mean)

    def _bias_one(r):
        cnt = r['count']
        if r['sum_y'] > 0:  # 有正样本
            return abs(weird_division(r['avg_p'], r['avg_y']) - 1) * cnt
        else:  # 全 0
            bayes = overall_alpha / (cnt + overall_alpha + overall_beta)
            return abs(weird_division(r['avg_p'], bayes) - 1) * cnt

    grouped['bias_val'] = grouped.apply(_bias_one, axis=1)

    return weird_division(grouped['bias_val'].sum(), len(y_true))


@convert_dataframe_to_numpy
@clip_predictions
def ece(y_true, y_pred, buckets=10, **kwargs):
    """
    Expected Calibration Error (ECE)
    """
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})

    df['bin_idx'] = pd.cut(df['y_pred'], bins=buckets, labels=False, include_lowest=True)

    bin_stats = df.groupby('bin_idx', observed=False).agg(
        n=('y_true', 'count'),
        acc=('y_true', 'mean'),
        conf=('y_pred', 'mean')
    ).reset_index()

    # 计算 |acc - conf| * 稀疏重
    bin_stats['weight'] = weird_division(bin_stats['n'], df.shape[0])
    bin_stats['contri'] = np.abs(bin_stats['acc'] - bin_stats['conf']) * bin_stats['weight']

    ece_val = bin_stats['contri'].sum()
    return ece_val


@convert_dataframe_to_numpy
@clip_predictions
def gece(y_true, y_pred, group_col, **kwargs):
    """
    Group-wise Expected Calibration Error (GECE)
    """
    grouped = gen_group_samples(y_true, y_pred, group_col)

    grp = grouped.agg(
        n=('y_true', 'count'),
        acc=('y_true', 'mean'),
        conf=('y_pred', 'mean')
    ).reset_index()

    # 计算 |acc - conf| * weight
    total = len(y_true)
    grp['weight'] = weird_division(grp['n'], total)
    grp['contri'] = np.abs(grp['acc'] - grp['conf']) * grp['weight']

    gece_val = grp['contri'].sum()
    return gece_val


@convert_dataframe_to_numpy
def mse(y_true, y_pred, **kwargs):
    """
    计算均方误差 (MSE)
    :param y_true: 真实值数组
    :param y_pred: 预测值数组
    :return: 均方误差
    """
    return np.mean(np.square(y_true - y_pred))


@convert_dataframe_to_numpy
def rise(y_true, y_pred, **kwargs):
    """
    计算均方根误差 (RISE)
    :param y_true: 真实值数组
    :param y_pred: 预测值数组
    :return: 均方根误差
    """
    return np.sqrt(np.mean(np.square(y_true - y_pred)))


@convert_dataframe_to_numpy
def mae(y_true, y_pred, **kwargs):
    """
    计算平均绝对误差 (MAE)
    :param y_true: 真实值数组
    :param y_pred: 预测值数组
    :return: 平均绝对误差
    """
    return np.mean(np.abs(y_true - y_pred))


@convert_dataframe_to_numpy
def r2_score(y_true, y_pred, **kwargs):
    """
    计算R方，衡量模型预测稳定性外差方差方向
    :param y_true: 真实值数组
    :param y_pred: 预测值数组
    :return: R方
    """
    y_true_mean = np.mean(y_true)
    ss_total = np.sum(np.square(y_true - y_true_mean))
    ss_residual = np.sum(np.square(y_true - y_pred))
    if ss_total == 0:  # 目标值全都相同时，视为完美，返回1
        return 1.0
    return 1 - (ss_residual / ss_total)


@convert_dataframe_to_numpy
def mape(y_true, y_pred, **kwargs):
    """
    计算 Mean Absolute Percentage Error（均方绝对百分比误差）
    """
    mask = y_true != 0
    if not np.any(mask):
        return np.nan
    mape_value = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]) * 100)
    return mape_value


@convert_dataframe_to_numpy
def pearson(y_true, y_pred, **kwargs):
    """
    计算 Pearson 相关系数，衡量模型预测趋势是否正确
    衡量预测和真实的线性相关性，越接近1越好
    """
    if len(y_true) < 2:
        return np.nan
    corr_matrix = np.corrcoef(y_true.flatten(), y_pred.flatten())
    return corr_matrix[0, 1]


@convert_dataframe_to_numpy
def explained_variance(y_true, y_pred, **kwargs):
    """
    计算 Explained Variance（解释方差），衡量模型预测稳定性和误差波动大小
    衡量模型解释了多少真实数据的方差，越接近1越好（波动小，预测稳定）
    """
    var_y_true = np.var(y_true)
    if var_y_true == 0:
        return 1.0
    var_residual = np.var(y_true - y_pred)
    evs = 1 - (var_residual / var_y_true)
    return evs


# ===== 分布相似度指标（无监督/生成模型适用）=====

def _value_freq_distributions(y_true, y_pred, top_k=None):
    """
    将两组原始样本值转为共同词表上的频率分布向量。
    :param y_true: 真实样本 np.ndarray (1-D)
    :param y_pred: 生成样本 np.ndarray (1-D)
    :param top_k: 若不为 None，只保留 y_true 中频次最高的 top_k 个值
    :return: p (真实频率), q (生成频率), 均为 np.ndarray，长度相同
    """
    real_vals, real_cnts = np.unique(y_true, return_counts=True)
    gen_vals, gen_cnts = np.unique(y_pred, return_counts=True)

    if top_k is not None:
        top_idx = np.argsort(real_cnts)[::-1][:top_k]
        real_vals = real_vals[top_idx]
        real_cnts = real_cnts[top_idx]

    vocab = real_vals  # 以真实分布的词表为基准
    n_real = len(y_true)
    n_gen = len(y_pred)

    # 建立 gen 的计数查找表
    gen_lookup = dict(zip(gen_vals, gen_cnts))

    p = real_cnts / n_real
    q = np.array([gen_lookup.get(v, 0) for v in vocab], dtype=np.float64) / n_gen

    return p, q


@convert_dataframe_to_numpy
def js_divergence(y_true, y_pred, top_k=None, **kwargs):
    """
    计算 Jensen-Shannon Divergence（衡量两个值分布的相似性，越小越相似，范围 [0, ln2]）
    先从原始样本值统计频率分布，再计算 JSD。适用于离散/类别特征。
    D_JS(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)，其中 M = 0.5 * (P + Q)
    :param top_k: 仅使用 y_true 中频次最高的 top_k 个值
    """
    y_true = np.array(y_true).flatten().astype(str)
    y_pred = np.array(y_pred).flatten().astype(str)

    p, q = _value_freq_distributions(y_true, y_pred, top_k=top_k)

    eps = 1e-12
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    m = 0.5 * (p + q)

    kl_p_m = np.sum(p * np.log(p / m))
    kl_q_m = np.sum(q * np.log(q / m))
    return float(0.5 * (kl_p_m + kl_q_m))


@convert_dataframe_to_numpy
def tv_distance(y_true, y_pred, top_k=None, **kwargs):
    """
    计算 Total Variation Distance（全变差距离，越小越相似，范围 [0, 1]）
    TV(P, Q) = 0.5 * sum |p_i - q_i|，先从原始样本值统计频率分布再计算。
    :param top_k: 仅使用 y_true 中频次最高的 top_k 个值
    """
    y_true = np.array(y_true).flatten().astype(str)
    y_pred = np.array(y_pred).flatten().astype(str)

    p, q = _value_freq_distributions(y_true, y_pred, top_k=top_k)
    return float(0.5 * np.sum(np.abs(p - q)))


@convert_dataframe_to_numpy
def ks_stat(y_true, y_pred, **kwargs):
    """
    计算 Kolmogorov-Smirnov 双样本统计量（衡量两个连续分布的最大偏差，越小越相似）
    纯 numpy 实现，无需 scipy。适用于数值型特征。
    对非数值型特征返回 NaN。
    """
    try:
        a = np.array(y_true, dtype=np.float64).flatten()
        b = np.array(y_pred, dtype=np.float64).flatten()
    except (ValueError, TypeError):
        return np.nan

    if len(a) == 0 or len(b) == 0:
        return np.nan

    a = np.sort(a)
    b = np.sort(b)
    n1, n2 = len(a), len(b)

    # 合并排序后的唯一值，在每个断点处比较 ECDF
    combined = np.concatenate([a, b])
    combined = np.unique(combined)

    cdf_a = np.searchsorted(a, combined, side='right') / n1
    cdf_b = np.searchsorted(b, combined, side='right') / n2

    return float(np.max(np.abs(cdf_a - cdf_b)))


# ===== Spearman 相关系数 =====
@convert_dataframe_to_numpy
def spearman_corr(y_true, y_pred, **kwargs):
    """
    计算 Spearman Rank Correlation（秩相关系数）
    衡量预测与真实的排序一致性，越接近1越好
    数学定义实现，无需 scipy
    """
    y_true = np.array(y_true, dtype=np.float64).flatten()
    y_pred = np.array(y_pred, dtype=np.float64).flatten()

    n = len(y_true)
    if n < 2:
        return np.nan

    # 计算秩（rank），平均处理重复值
    def rankdata(x):
        tmp = np.argsort(x)
        ranks = np.empty_like(tmp, dtype=float)
        ranks[tmp] = np.arange(len(x))
        # 对重复值取平均秩
        _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
        avg_ranks = np.bincount(inv, ranks) / counts
        return avg_ranks[inv]

    rank_true = rankdata(y_true)
    rank_pred = rankdata(y_pred)

    # 使用皮尔逊相关系数计算秩相关
    corr_matrix = np.corrcoef(rank_true, rank_pred)
    return float(corr_matrix[0, 1])


@convert_dataframe_to_numpy
def grouped_mse(y_true, y_pred, group_col):
    """
    计算分组均方误差 (Grouped MSE)
    :param y_true: 真实值数组
    :param y_pred: 预测值数组
    :param group_col: 分组列数组
    :return: 分组均方误差
    """
    if len(y_true) != len(y_pred) or len(y_true) != len(group_col):
        raise ValueError("Input arrays must have the same length")

    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'group_col': group_col
    })

    group_mse = df.groupby('group_col').apply(
        lambda group: np.mean(np.square(group['y_true'] - group['y_pred']))
    ).reset_index(name='mse')

    unique_groups = df['group_col'].unique()
    result = np.zeros(len(unique_groups))
    for i, group in enumerate(unique_groups):
        mask = group_mse['group_col'] == group
        if mask.any():
            result[i] = group_mse.loc[mask, 'mse'].values[0]
        else:
            result[i] = 0.0
    return result


@convert_dataframe_to_numpy
def event_recall(y_true, y_pred, **kwargs):
    """
    二分类的召回率
    y_true, y_pred: {0, 1}
    """
    y_true_event = (y_true != 0).astype(int)
    y_pred_event = (y_pred != 0).astype(int)

    tp = np.sum((y_true_event == 1) & (y_pred_event == 1))
    fn = np.sum((y_true_event == 1) & (y_pred_event == 0))

    if tp + fn == 0:
        return np.nan
    return tp / (tp + fn)


@convert_dataframe_to_numpy
def event_precision(y_true, y_pred, **kwargs):
    """
    Event Precision:
    在预测为发生反馈的样本中，真实发生的比例
    """
    y_true_event = (y_true != 0).astype(int)
    y_pred_event = (y_pred != 0).astype(int)

    tp = np.sum((y_true_event == 1) & (y_pred_event == 1))
    fp = np.sum((y_true_event == 0) & (y_pred_event == 1))

    if tp + fp == 0:
        return np.nan
    return tp / (tp + fp)


@convert_dataframe_to_numpy
def accuracy(y_true, y_pred, **kwargs):
    if len(y_true) == 0:
        return np.nan
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    return np.mean(y_true == y_pred)
