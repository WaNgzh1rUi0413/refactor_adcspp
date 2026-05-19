import logging
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd

from inputs.dataset_builder import DatasetBuilder
from .abstract_task import AbstractTask
from .evaluate import Evaluate
from utils.config.constants import MTP_DATA_DIR, MTP_OUTPUT_DIR
from utils.easydict import EasyDict
from utils.global_variables import MTP_TASK_CONFIG
from .train import FLOW_RUNTIME_CACHE

_DEFAULT_GENERATIVE_METRICS = [
    {"type": "js_divergence", "name": "JS"},
    {"type": "tv_distance", "name": "TV"},
    {"type": "ks_stat", "name": "KS"},
]


class PredictTask(AbstractTask):
    step = "predict"
    self_managed = True

    def __init__(self, cfg):
        self.cfg = cfg

    @staticmethod
    def _merge_cfg(base_cfg, override_cfg):
        base = dict(base_cfg) if base_cfg is not None else {}
        if override_cfg is not None:
            base.update(dict(override_cfg))
        return EasyDict(base)

    @staticmethod
    def _expand_path_vars(path: str) -> str:
        if not path:
            return ""
        mappings = {
            "$dataDir": MTP_DATA_DIR,
            "${dataDir}": MTP_DATA_DIR,
            "$outputDir": MTP_OUTPUT_DIR,
            "${outputDir}": MTP_OUTPUT_DIR,
            "$scene_name": MTP_TASK_CONFIG.scene_name,
            "${scene_name}": MTP_TASK_CONFIG.scene_name,
            "$model_id": MTP_TASK_CONFIG.model_id,
            "${model_id}": MTP_TASK_CONFIG.model_id,
            "$date": MTP_TASK_CONFIG.date,
            "${date}": MTP_TASK_CONFIG.date,
        }
        expanded = os.path.expandvars(path)
        for key, value in mappings.items():
            expanded = expanded.replace(key, value or "")
        return os.path.normpath(expanded)

    def _resolve_output_path(self, output_dir: Path, file_name: str) -> Path:
        expanded_file = self._expand_path_vars(file_name)
        target = Path(expanded_file)
        if target.is_absolute():
            return target
        return output_dir / target

    @staticmethod
    def _ensure_parent_dir(file_path: Path) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _pick_eval_df(test_df, valid_df, real_df):
        if test_df is not None:
            return "test", test_df
        if valid_df is not None:
            return "valid", valid_df
        return "train", real_df

    def _load_splits(self, input_cfg, schema):
        input_cfg.schema = schema
        dataset = DatasetBuilder.build(input_cfg)
        try:
            datasets = dataset.build(["train", "valid", "test"])
        except ValueError as exc:
            msg = str(exc).lower()
            if "split=test" not in msg:
                raise
            logging.info("Test split not found, fallback to [train, valid]: %s", exc)
            datasets = dataset.build(["train", "valid"])

        real_df = datasets[0]
        valid_df: Optional[object] = None
        test_df: Optional[object] = None
        if len(datasets) > 1 and len(datasets[1]) > 0:
            valid_df = datasets[1]
        if len(datasets) > 2 and len(datasets[2]) > 0:
            test_df = datasets[2]
        return real_df, valid_df, test_df

    def _build_evaluator(self) -> Evaluate:
        """从配置构建 Evaluate，若未配置则使用默认的分布类指标。"""
        evaluator_cfg = self.cfg.get("evaluator", {})
        if not evaluator_cfg.get("metric"):
            evaluator_cfg = dict(evaluator_cfg)
            evaluator_cfg["metric"] = _DEFAULT_GENERATIVE_METRICS
        evaluator_cfg.setdefault("y_true", "y_true")
        evaluator_cfg.setdefault("y_pred", "y_pred")
        return Evaluate(evaluator_cfg)

    def _evaluate_distributions(
        self,
        real_df: pd.DataFrame,
        gen_df: pd.DataFrame,
        evaluator: Evaluate,
        columns: List[str],
    ) -> pd.DataFrame:
        """
        对每个特征列分别调用 Evaluate，比较真实数据与生成数据的分布。
        两个 DataFrame 行数可以不同；取较短的一侧对齐后传入指标函数。
        :return: DataFrame，index 为 feature，列为各指标值
        """
        n = min(len(real_df), len(gen_df))
        rows = []
        for col in columns:
            if col not in real_df.columns or col not in gen_df.columns:
                logging.debug("Column %s missing in real or gen df, skipped", col)
                continue
            col_df = pd.DataFrame({
                evaluator.y_true_key: real_df[col].iloc[:n].values,
                evaluator.y_pred_key: gen_df[col].iloc[:n].values,
            })
            metrics = evaluator._calculate_metrics(col_df, dataset_name=col)
            rows.append({"feature": col, **metrics})
        return pd.DataFrame(rows)

    def run(self) -> None:
        train_cfg = FLOW_RUNTIME_CACHE.get("cfg")
        model = FLOW_RUNTIME_CACHE.get("model")
        real_df = FLOW_RUNTIME_CACHE.get("train_df")
        valid_df = FLOW_RUNTIME_CACHE.get("valid_df")
        test_df = FLOW_RUNTIME_CACHE.get("test_df")

        if model is None:
            raise ValueError("PredictTask requires TrainTask to run first in the same pipeline run")
        train_input = train_cfg.input if train_cfg is not None and train_cfg.get("input") else None
        train_output = train_cfg.output if train_cfg is not None and train_cfg.get("output") else None
        input_cfg = self._merge_cfg(train_input, self.cfg.get("input"))
        output_cfg = self._merge_cfg(train_output, self.cfg.get("output"))
        schema = self.cfg.get("schema") or (train_cfg.schema if train_cfg is not None else None)

        if not input_cfg or not output_cfg or schema is None:
            raise ValueError("PredictTask requires input/output/schema or a preceding TrainTask config")

        if real_df is None:
            real_df, valid_df, test_df = self._load_splits(input_cfg, schema)

        n_gen = int(self.cfg.n_gen)
        target_timehour = self.cfg.target_timehour
        gen_df = model.generate(n=n_gen, target_timehour=target_timehour)

        output_dir = Path(self._expand_path_vars(input_cfg.get("output_dir", "output")))
        output_dir.mkdir(parents=True, exist_ok=True)

        generated_csv = self._resolve_output_path(output_dir, output_cfg.generated_csv)
        self._ensure_parent_dir(generated_csv)
        gen_df.to_csv(generated_csv, index=False, encoding="utf-8-sig")
        logging.info("Saved generated data to %s", os.path.abspath(generated_csv))

        eval_name, eval_df = self._pick_eval_df(test_df, valid_df, real_df)
        logging.info("Evaluation dataset split: %s (rows=%d)", eval_name, len(eval_df))

        evaluator = self._build_evaluator()
        eval_cols = [col for col in gen_df.columns if col in eval_df.columns]
        results_df = self._evaluate_distributions(eval_df, gen_df, evaluator, eval_cols)
        logging.info("Distribution evaluation results:\n%s", results_df.to_string(index=False))

        eval_results_name = output_cfg.get("eval_results_csv", "eval_distribution_results.csv")
        eval_results_path = self._resolve_output_path(output_dir, eval_results_name)
        self._ensure_parent_dir(eval_results_path)
        results_df.to_csv(eval_results_path, index=False, encoding="utf-8-sig")
        logging.info("Saved distribution evaluation results to %s", os.path.abspath(eval_results_path))
