import logging
import numbers
import os.path
import time
from typing import Dict, Any

import pandas as pd
import numpy as np

from core.registry import FunctionRegistry
from utils.global_variables import PATH_CONFIG
from utils.io import get_specified_feature_map_dict
from utils.verify import validate_dataframe_function


class EvaluateFuncRegistry(FunctionRegistry):
    """Evaluate function registry."""

    def __init__(self):
        super().__init__(
            registry_name="EvaluateFuncRegistry",
            validator=validate_dataframe_function,
        )
        # Auto register metric functions
        self.auto_register_from_module(module_path="metrics.metric_funcs")


class Evaluate:
    def __init__(self, evaluator_config: Dict[str, Any]):
        self.metrics = evaluator_config.get("metric", [])
        self.metric_group_cols = evaluator_config.get("metric_group_cols", [])
        self.evaluate_func_registry = EvaluateFuncRegistry()
        self.y_true_key = evaluator_config.get("y_true", "label")
        self.y_pred_key = evaluator_config.get("y_pred", "pred")
        self.name = evaluator_config.get("name", f"{self.y_true_key}_{self.y_pred_key}")
        self.cfg = evaluator_config
        self.save_evaluator = evaluator_config.get("save_evaluator", False)
        if self.save_evaluator:
            self.model_id = evaluator_config.get("model_id", "")
            logging.info(f"model_id:{self.model_id}")
            self.date = evaluator_config.get("date", "")

    def run(self, y_true_dict, y_pred_dict, group_columns=None):
        """
        Calculate metrics.

        :param y_true_dict: label Dict[str, np.ndarray[N,1]]
        :param y_pred_dict: pred Dict[str, np.ndarray[N,1]]
        :param group_columns: Dict[str, np.ndarray[N,1]]
        :return: metrics dict
        """
        if group_columns is None:
            group_columns = {}

        eval_df = pd.DataFrame({
            **y_true_dict,
            **y_pred_dict,
            **group_columns,
        })

        results = {"global": self._calculate_metrics(eval_df, dataset_name="global")}

        if self.save_evaluator:
            self.save_evaluator_file(eval_df, group_name="all")

        for group in self.metric_group_cols:
            query_conditions, group_name_parts = self._get_group_conditions(
                group, eval_df, y_true_dict, y_pred_dict
            )

            if query_conditions:
                group_name = "_AND_".join(group_name_parts)
                combined_query = " & ".join(query_conditions)
                subgroup_df = eval_df.query(combined_query, engine="python")

                if not subgroup_df.empty:
                    results[group_name] = self._calculate_metrics(subgroup_df, group_name)
                    if self.save_evaluator:
                        self.save_evaluator_file(subgroup_df, group_name)
                else:
                    if self.save_evaluator:
                        self.save_evaluator_file(subgroup_df, group_name)
                    else:
                        results[group_name] = {
                            k: None for k in results.get("global", {}).keys()
                        }

        self._save_monitor_file(self.y_true_key, results)
        self._save_result_file(results)

        results = self._format_result_to_table(results)

        return results

    def _get_group_conditions(self, group, eval_df, y_true_dict, y_pred_dict):
        """Generate group filtering conditions."""
        query_conditions = []
        group_name_parts = []

        for col, values in group.items():
            if not (isinstance(values, list) and col in eval_df.columns):
                continue

            if col in y_true_dict or col in y_pred_dict:
                group_name_parts.append(f"{col}={'-'.join(map(str, values))}")
                condition = (
                    f"`{col}` in {values}"
                    if len(values) > 1
                    else f"`{col}` == {values[0]}"
                )
                query_conditions.append(condition)
                continue

            featuremap_dict = get_specified_feature_map_dict(col)
            group_name_parts.append(f"{col}={'-'.join(map(str, values))}")
            values = [featuremap_dict.get(v) for v in values]
            condition = (
                f"`{col}` in {values}"
                if len(values) > 1
                else f"`{col}` == {values[0]}"
            )
            query_conditions.append(condition)

        return query_conditions, group_name_parts

    @staticmethod
    def _save_monitor_file(label_name, results):
        """
        Save monitor file.

        :param results: metric dict
        """
        monitor_csv_path = PATH_CONFIG.monitor_csv_path

        monitor_df = pd.DataFrame.from_dict(results, orient="index")
        monitor_df = monitor_df.reset_index().rename(columns={"index": "scenario"})

        monitor_df["y_true"] = label_name
        if os.path.exists(monitor_csv_path):
            existing_df = pd.read_csv(monitor_csv_path)
            existing_df = existing_df[existing_df["y_true"] != label_name]
            final_df = pd.concat([existing_df, monitor_df], ignore_index=True)
        else:
            final_df = monitor_df

        final_df.to_csv(monitor_csv_path, index=False, float_format="%.6f")

    def _save_result_file(self, results):
        target_path = self.cfg.get("result_file_path") or PATH_CONFIG.result_file_path
        pairs = []
        for scene, sub in results.items():
            for k, v in sub.items():
                if not isinstance(v, numbers.Number):
                    logging.warning(
                        "save_result_file skip %s of %s because it is not a number",
                        k,
                        scene,
                    )
                    continue
                if scene == "global":
                    pairs.append(f"{k}:{v}")
                else:
                    pairs.append(f"{k}_{scene}:{v}")

        line = ";".join(pairs)

        with open(target_path, "w", encoding="utf-8") as f:
            f.write(line)

    def _calculate_metrics(self, eval_df, dataset_name=""):
        """Internal calculation method."""
        results = {}
        for metric in self.metrics:
            tag = metric["type"]
            start_time = time.time()
            func = self.evaluate_func_registry.get(tag)
            if not func:
                logging.warning("[%s-%s] skip metric not supported", dataset_name, tag)
                continue

            config = metric.get("config", {})
            try:
                value = func(
                    eval_df=eval_df,
                    y_true_key=self.y_true_key,
                    y_pred_key=self.y_pred_key,
                    **config,
                )
                results[metric.get("name", tag)] = (
                    round(value, 6) if isinstance(value, float) else value
                )
            except Exception as e:
                logging.warning(
                    "[%s-%s] metric calc failed: %s",
                    dataset_name,
                    tag,
                    str(e),
                    exc_info=True,
                )
                results[metric.get("name", tag)] = None
            end_time = time.time()
            logging.info(
                "finished calculation of metrics %s of %s, cost: %.2f s, value: %s",
                tag,
                dataset_name,
                end_time - start_time,
                results.get(metric.get("name", tag)),
            )

        return results

    @staticmethod
    def _format_result_to_table(data):
        all_groups = data.copy()
        result = "\n"
        metrics_names = list(all_groups["global"].keys())

        def format_value(value):
            if value is None:
                return "nan"
            if isinstance(value, float):
                return f"{value:.6f}"
            return str(value)

        max_widths = {name: len(name) for name in metrics_names}
        max_widths["group"] = max([len(key) for key in all_groups] + [len("group")])

        for _, metrics in all_groups.items():
            for name in metrics_names:
                value = metrics.get(name)
                formatted = format_value(value)
                max_widths[name] = max(max_widths[name], len(formatted))

        header_parts = []
        for name in ["group"] + metrics_names:
            width = max_widths["group"] if name == "group" else max_widths[name]
            header_parts.append(
                f"{name:<{width}}" if name == "group" else f"{name:>{width}}"
            )
        header_line = " | ".join(header_parts)

        line_parts = [f"{'-' * max_widths['group']}"] + [
            f"{'-' * max_widths[name]}" for name in metrics_names
        ]
        separator_line = "-+-".join(line_parts)

        result += header_line + "\n"
        result += separator_line + "\n"

        if "global" in all_groups:
            global_metrics = all_groups.pop("global")
            row_parts = [f"{'global':<{max_widths['group']}}"]
            for name in metrics_names:
                value = global_metrics.get(name)
                formatted = format_value(value)
                row_parts.append(f"{formatted:>{max_widths[name]}}")
            result += " | ".join(row_parts) + "\n"
            result += separator_line + "\n"

        for group, metrics in all_groups.items():
            row_parts = [f"{group:<{max_widths['group']}}"]
            for name in metrics_names:
                value = metrics.get(name)
                formatted = format_value(value)
                row_parts.append(f"{formatted:>{max_widths[name]}}")
            result += " | ".join(row_parts) + "\n"

        return result

    def save_evaluator_file(self, dictionary, group_name):
        """
        Save label/pred histogram file.

        :param dictionary: data dict
        :param group_name: saved group name
        """

        def save_file(base_path, dictionary, group_name, key):
            # Ensure base path exists
            if not os.path.exists(base_path):
                os.makedirs(base_path)

            file_path = os.path.join(base_path, f"{key}_{group_name}.npy")

            with open(file_path, "wb") as file:
                value = dictionary[key].values
                np.save(file, value)

            logging.info(f"{key}_{group_name}_histogram")

        model_id = str(self.model_id)
        date = str(self.date)

        base_path = os.path.join(
            "/opt/huawei/dataset/dataDir/outputfiles",
            model_id,
            "analysis",
            date,
        )

        save_file(base_path, dictionary, group_name, self.y_true_key)
        save_file(base_path, dictionary, group_name, self.y_pred_key)
