"""TFRecord-backed AdCSPP behavior training and prediction tasks."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import tensorflow as tf

from inputs.adcsp.tfrecord_adapter import AdCSPPTFRecordAdapter, batch_size
from models.adcsp.model import AdCSPPModel
from utils import global_variables as gv
from utils.easydict import EasyDict
from .abstract_task import AbstractTask


def _root_config() -> Mapping[str, Any]:
    return getattr(gv, "TOTAL_CONFIG", {}) or {}


class _AdCSPPTaskBase(AbstractTask):
    self_managed = False
    uses_custom_model = True

    def __init__(self, cfg: EasyDict | None = None):
        self.cfg = cfg or EasyDict()
        self.result = None

    def _config(self, name: str, default=None):
        if name in self.cfg:
            return self.cfg.get(name)
        return _root_config().get(name, default)

    def _model(self) -> AdCSPPModel:
        model_cfg = self._config("model", {}) or {}
        return AdCSPPModel(
            latent_dim=int(model_cfg.get("latent_dim", 16)),
            static_dim=int(model_cfg.get("static_dim", 16)),
            context_dim=int(model_cfg.get("context_dim", 16)),
            hidden_dim=int(model_cfg.get("hidden_dim", 32)),
            seed=int(model_cfg.get("seed", 42)),
            feature_config=model_cfg.get("feature_config", {}),
        )

    def _adapter(self) -> AdCSPPTFRecordAdapter:
        model_cfg = self._config("model", {}) or {}
        return AdCSPPTFRecordAdapter(model_cfg.get("feature_mapping", {}))

    @staticmethod
    def _restore(model: AdCSPPModel, checkpoint_path: str | None) -> None:
        if checkpoint_path and Path(checkpoint_path).exists():
            model.load_weights(checkpoint_path)


class AdCSPPFlowTrainTask(_AdCSPPTaskBase):
    """Train click and post-click conversion behavior from platform TFRecords."""

    step = "train"

    def run(self) -> None:
        if self.train_dataset is None:
            raise ValueError("AdCSPPFlowTrainTask requires the platform train_dataset")
        model_cfg = self._config("model", {}) or {}
        model = self._model()
        adapter = self._adapter()
        optimizer = tf.keras.optimizers.Adam(learning_rate=float(self.cfg.get("learning_rate", model_cfg.get("learning_rate", 1e-3))))
        conversion_weight = float(self.cfg.get("conversion_loss_weight", 2.0))
        click_window = int(self.cfg.get("click_window_seconds", 30))
        conversion_window = int(self.cfg.get("conversion_window_seconds", 3600))
        epochs = int(self.cfg.get("epochs", 1))
        last_metrics = {}

        for _ in range(epochs):
            with tf.GradientTape() as tape:
                click_probabilities, conversion_probabilities = [], []
                click_labels, conversion_labels = [], []
                for batch in self.train_dataset:
                    features, labels = batch[:2]
                    for index in range(batch_size(features, labels)):
                        example = adapter.example(features, labels, index, latent_dim=int(model_cfg.get("latent_dim", 16)), feature_config=model_cfg.get("feature_config", {}))
                        click_probability, conversion_probability = model.differentiable_probabilities(example.state, example.context, click_window, conversion_window)
                        click_probabilities.append(tf.reshape(click_probability, []))
                        conversion_probabilities.append(tf.reshape(conversion_probability, []))
                        click_labels.append(example.is_click)
                        conversion_labels.append(example.is_conversion)

                if not click_probabilities:
                    raise ValueError("AdCSPPFlowTrainTask received an empty train dataset")
                click_probability = tf.stack(click_probabilities)
                conversion_probability = tf.stack(conversion_probabilities)
                click_label = tf.convert_to_tensor(click_labels, dtype=tf.float32)
                conversion_label = tf.convert_to_tensor(conversion_labels, dtype=tf.float32)
                click_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(click_label, click_probability))
                conversion_element_loss = tf.keras.losses.binary_crossentropy(conversion_label, conversion_probability)
                click_mask = tf.stop_gradient(click_label)
                conversion_loss = tf.reduce_sum(conversion_element_loss * click_mask) / tf.maximum(tf.reduce_sum(click_mask), 1.0)
                total_loss = click_loss + conversion_weight * conversion_loss

            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients((gradient, variable) for gradient, variable in zip(gradients, model.trainable_variables) if gradient is not None)
            last_metrics = {"loss": float(total_loss.numpy()), "click_loss": float(click_loss.numpy()), "conversion_loss": float(conversion_loss.numpy())}

        checkpoint_path = self.cfg.get("checkpoint_path", model_cfg.get("checkpoint_path", "checkpoints/adcsp_model.weights.h5"))
        checkpoint = Path(checkpoint_path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        model.built = True
        model.save_weights(str(checkpoint))
        self.result = {"checkpoint_path": str(checkpoint), **last_metrics}
        logging.info("AdCSPP behavior training finished: %s", self.result)


class AdCSPPFlowPredictTask(_AdCSPPTaskBase):
    """Predict click and conversion behavior from platform test TFRecords."""

    step = "predict"

    def run(self) -> None:
        if self.test_dataset is None:
            raise ValueError("AdCSPPFlowPredictTask requires the platform test_dataset")
        model_cfg = self._config("model", {}) or {}
        model = self._model()
        adapter = self._adapter()
        click_window = int(self.cfg.get("click_window_seconds", 30))
        conversion_window = int(self.cfg.get("conversion_window_seconds", 3600))
        outputs = {"pred_click_probability": [], "pred_conversion_probability": []}
        restored = False
        for batch in self.test_dataset:
            features, labels = batch[:2]
            for index in range(batch_size(features, labels)):
                example = adapter.example(features, labels, index, latent_dim=int(model_cfg.get("latent_dim", 16)), feature_config=model_cfg.get("feature_config", {}))
                if not restored:
                    # Build all Keras variables before restoring the checkpoint.
                    model.differentiable_probabilities(example.state, example.context, click_window, conversion_window)
                    model.built = True
                    self._restore(model, self.cfg.get("checkpoint_path", model_cfg.get("checkpoint_path")))
                    restored = True
                outputs["pred_click_probability"].append(model.click_probability(example.state, example.context, click_window))
                outputs["pred_conversion_probability"].append(model.conversion_probability(example.state, example.context, conversion_window))
        self.results = outputs
        self.result = {"record_count": len(outputs["pred_click_probability"]), "outputs": outputs}
        logging.info("AdCSPP behavior prediction finished: %d records", self.result["record_count"])

