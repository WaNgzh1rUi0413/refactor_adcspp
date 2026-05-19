import argparse
import logging
import os
import re
from typing import List, Tuple

USE_NPU = os.environ.get('useNpuTraining', 'false') == 'true'
if USE_NPU:
    try:
        # 需要在导入tf之前执行，避免有tf op/tensor未迁移到NPU
        import npu_device as npu
        # 默认设置Device 0为计算设备
        npu.open().as_default()
    except Exception as e:
        logging.error('Set NPU failed. Error message:\n%s', e)
        logging.warning('Use CPU for training.')
    else:
        logging.warning('Use NPU for training.')

import tensorflow as tf

from tasks.pipeline_builder import PipelineBuilder
from utils import global_variables as gv
from utils.config.config_utils import get_cfg_from_file
from utils.easydict import EasyDict
from utils.logger import set_logger


def ensure_easydict_section(cfg: EasyDict, section: str) -> EasyDict:
    """Ensure cfg.<section> exists and is EasyDict."""
    value = cfg.get(section)
    if isinstance(value, EasyDict):
        return value
    if isinstance(value, dict):
        value = EasyDict(value)
    else:
        value = EasyDict()
    setattr(cfg, section, value)
    return value


def parse_args() -> Tuple[argparse.Namespace, List[str]]:
    """
    解析命令行参数，返回解析后的Namespace实例
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', '--configPath', type=str, help='path to config file')
    parser.add_argument('--date', type=str, default='', help='running period')
    parser.add_argument('--gpu_device', '--gpuDevice', '--useGpu', dest='gpu_device', type=str, default='',
                        help='running period')
    parser.add_argument('--step', type=str, default='', help='steps to run, separated by ,')
    parser.add_argument('--scene_name', '--sceneName', '--scenName', dest='scene_name', default='', type=str)
    parser.add_argument('--version', default='not_use', type=str)
    parser.add_argument('--model_id', '--modelId', dest='model_id', default='', type=str)
    parser.add_argument('--period', type=str, default='', help='period')

    args, unknown_args = parser.parse_known_args()
    return args, unknown_args


def main(cfg: EasyDict) -> None:
    """
    入口函数
    """
    pipeline = PipelineBuilder.build(cfg)
    pipeline.run()


def parse_step_override(raw_step: str) -> List[str]:
    if not raw_step:
        return []
    parts = [s.strip() for s in re.split(r"[\s,\-]+", raw_step) if s.strip()]
    return parts


def query_and_set_gpu() -> int:
    """
    设置GPU按需分配内存
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # 设置为按需分配内存
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as rte:
            # 内存增长设置必须在程序启动时进行
            logging.error('Error: %s', rte)
    return len(gpus)


if __name__ == '__main__':
    import json

    args, unknown_args = parse_args()
    cfg = get_cfg_from_file(args.config_path, merge_default=False)

    set_logger(cfg.run.log_level)
    num_gpus = query_and_set_gpu()

    # 命令行中step会覆盖config中配置的step
    if args.step:
        parsed_steps = parse_step_override(args.step)
        if parsed_steps:
            cfg.run.steps = parsed_steps

    logging.info('Effective run.steps: %s', cfg.run.steps)

    run_cfg = ensure_easydict_section(cfg, 'run')
    predict_cfg = ensure_easydict_section(cfg, 'predict')

    run_cfg.num_device = num_gpus
    run_cfg.num_used_device = args.gpu_device
    run_cfg.use_npu = USE_NPU
    predict_cfg.model_id = args.model_id
    predict_cfg.date = args.date
    predict_cfg.period = args.period

    if unknown_args:
        logging.warning('Ignore unknown cli args from platform: %s', unknown_args)

    logging.info('Total config is %s', json.dumps(cfg, indent=4))
    gv.TOTAL_CONFIG = cfg
    gv.MTP_TASK_CONFIG.set_value(args.date, args.version, args.scene_name, args.model_id)
    logging.info('MTP_TASK_CONFIG is %s', gv.MTP_TASK_CONFIG)
    main(cfg)
