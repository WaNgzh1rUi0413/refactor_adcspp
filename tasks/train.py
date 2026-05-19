import logging
import os
import json
import inspect
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf

from inputs.dataset_builder import DatasetBuilder
from models.model_builder import ModelBuilder
from core.model_export import export_onnx
from utils.config.constants import MTP_DATA_DIR, MTP_OUTPUT_DIR
from utils.global_variables import MTP_TASK_CONFIG, PATH_CONFIG
from .abstract_task import AbstractTask


FLOW_RUNTIME_CACHE = {}


class TrainTask(AbstractTask):
    step = "train"
    self_managed = True

    def __init__(self, cfg):
        self.cfg = cfg
        self.checkpoint_path: Optional[str] = None
        self.model_export_path: Optional[str] = None

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

    def _resolve_output_path(self, output_dir: str, file_name: str) -> str:
        expanded_file = self._expand_path_vars(file_name)
        if os.path.isabs(expanded_file):
            return expanded_file
        return os.path.join(output_dir, expanded_file)

    @staticmethod
    def _ensure_parent_dir(path: str) -> None:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    @staticmethod
    def _safe_model_save(model, save_path: str, include_optimizer: bool = False) -> None:
        save_fn = getattr(model, "save", None)
        if not callable(save_fn):
            raise AttributeError("Model has no callable save()")

        try:
            sig = inspect.signature(save_fn)
            if "include_optimizer" in sig.parameters:
                save_fn(save_path, include_optimizer=include_optimizer)
            else:
                save_fn(save_path)
        except (TypeError, ValueError):
            save_fn(save_path)

    def _build_model_paths(self) -> None:
        output_cfg = self.cfg.get("output", {})
        input_output_dir = self.cfg.input.get("output_dir", "output")
        output_dir = self._expand_path_vars(input_output_dir)

        checkpoint_name = output_cfg.get("model_checkpoint")
        if checkpoint_name:
            self.checkpoint_path = self._resolve_output_path(output_dir, checkpoint_name)
        else:
            self.checkpoint_path = os.path.join(PATH_CONFIG.checkpoint_path, "generation_model.pt")

        export_name = output_cfg.get("model_export")
        if export_name:
            self.model_export_path = self._resolve_output_path(output_dir, export_name)
        else:
            self.model_export_path = os.path.join(PATH_CONFIG.model_export_path, "model_export")

        self.checkpoint_path = self._expand_path_vars(self.checkpoint_path)
        self.model_export_path = self._expand_path_vars(self.model_export_path)
        logging.info(
            "Checkpoints will go to %s. Exported model will be saved at %s",
            self.checkpoint_path,
            self.model_export_path,
        )

    def _save_model_version_file(self) -> None:
        logging.info("Start saving model version file")
        try:
            save_model_version_path = PATH_CONFIG.model_version_path
            if not os.path.exists(save_model_version_path):
                os.makedirs(save_model_version_path)

            version_file = os.path.join(save_model_version_path, "model_version.csv")
            version_data = {}
            if os.path.exists(version_file):
                with open(version_file, "r", encoding="utf-8") as file:
                    json_str = file.readline().strip()
                    if json_str:
                        version_data = json.loads(json_str)

            version_data[MTP_TASK_CONFIG.date] = self.checkpoint_path
            with open(version_file, "w", encoding="utf-8", newline="") as file:
                file.write(json.dumps(version_data, ensure_ascii=False))

            logging.info("Saved model version file to %s", version_file)
        except Exception as exc:
            logging.error("Save model version file failed: %s", exc)

    @staticmethod
    def _save_score_file(model, save_path: str, dataset) -> None:
        if dataset is None:
            return
        predict_fn = getattr(model, "predict", None)
        if not callable(predict_fn):
            logging.warning("Skip score export because model has no predict()")
            return

        try:
            target_path = save_path.replace("#period_date_placeholder", MTP_TASK_CONFIG.date or "")
            parent = os.path.dirname(target_path)
            if parent and (not os.path.exists(parent)):
                os.makedirs(parent)

            model_out = predict_fn(dataset)
            if isinstance(model_out, dict):
                score_data = {}
                for key, value in model_out.items():
                    arr = np.array(value)
                    if arr.ndim > 1:
                        arr = arr.reshape(arr.shape[0], -1)[:, 0]
                    score_data[key] = arr
                score_df = pd.DataFrame(score_data)
            else:
                arr = np.array(model_out)
                if arr.ndim > 1:
                    arr = arr.reshape(arr.shape[0], -1)[:, 0]
                score_df = pd.DataFrame({"score": arr})

            score_df.to_csv(target_path, index=False)
            logging.info("Saved score file to %s", os.path.abspath(target_path))
        except Exception as exc:
            logging.error("Save score file failed: %s", exc)

    def _export_model(self, model, real_df, valid_df) -> None:
        if self.cfg.get("save_best", False) and hasattr(model, "load_weights") and self.checkpoint_path:
            if os.path.exists(self.checkpoint_path):
                try:
                    model.load_weights(self.checkpoint_path)
                    logging.info("Loaded best checkpoint from %s before export", self.checkpoint_path)
                except Exception as exc:
                    logging.warning("Load checkpoint before export failed: %s", exc)
            else:
                logging.warning("save_best is true, but checkpoint does not exist: %s", self.checkpoint_path)

        if self.model_export_path:
            self._ensure_parent_dir(self.model_export_path)
            self._safe_model_save(model, self.model_export_path, include_optimizer=False)
            logging.info("Saved exported model to %s", os.path.abspath(self.model_export_path))

        if self.cfg.get("save_model_train_best_score", False):
            train_score_path = self.cfg.get("save_model_train_best_train_score_path", "")
            valid_score_path = self.cfg.get("save_model_train_best_valid_score_path", "")
            if train_score_path:
                self._save_score_file(model, train_score_path, real_df)
            if valid_score_path:
                self._save_score_file(model, valid_score_path, valid_df)

        if self.cfg.get("is_use_model_version", False):
            self._save_model_version_file()

        if self.cfg.get("save_onnx", False):
            if isinstance(model, tf.keras.Model):
                try:
                    export_onnx(model, self.model_export_path, self.cfg.get("remove_dummy_opset", True))
                except Exception as exc:
                    logging.error("Export onnx failed: %s", exc)
            else:
                logging.warning("save_onnx is true, but current model is not a tf.keras.Model")

    def _load_splits(self, dataset):
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

    def run(self) -> None:
        input_cfg = self.cfg.input
        input_cfg.schema = self.cfg.schema

        dataset = DatasetBuilder.build(input_cfg)
        real_df, valid_df, test_df = self._load_splits(dataset)
        logging.info("Loaded train rows: %d", len(real_df))
        if valid_df is not None:
            logging.info("Loaded valid rows: %d", len(valid_df))
        else:
            logging.info("Loaded valid rows: 0 (fallback to internal split)")
        if test_df is not None:
            logging.info("Loaded test rows: %d", len(test_df))
        else:
            logging.info("Loaded test rows: 0 (fallback to valid/train for evaluation)")

        model_cfg = self.cfg
        model_cfg.type = model_cfg.model.get("type", "generation_model")
        model = ModelBuilder.build(model_cfg)
        model.fit(real_df, df_valid=valid_df)

        self._build_model_paths()
        if hasattr(model, "save"):
            self._ensure_parent_dir(self.checkpoint_path)
            self._safe_model_save(model, self.checkpoint_path)
            logging.info("Saved model checkpoint to %s", os.path.abspath(self.checkpoint_path))

            # Preserve train_origin behavior: export model artifacts after training.
            self._export_model(model, real_df, valid_df)
        else:
            logging.warning("Model has no save(); skip checkpoint/export step")

        FLOW_RUNTIME_CACHE["model"] = model
        FLOW_RUNTIME_CACHE["train_df"] = real_df
        FLOW_RUNTIME_CACHE["valid_df"] = valid_df
        FLOW_RUNTIME_CACHE["test_df"] = test_df
        FLOW_RUNTIME_CACHE["cfg"] = self.cfg
        logging.info("Cached train artifacts for PredictTask")
