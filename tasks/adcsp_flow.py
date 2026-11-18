"""TFRecord-backed AdCSPP behavior training and prediction tasks."""

from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import tensorflow as tf

from inputs.adcsp.tfrecord_adapter import AdCSPPTFRecordAdapter, batch_size
from models.adcsp.state import UserStateBuilder
from models.adcsp.model import AdCSPPModel
from models.adcsp.generator import AdCSPPPathGenerator
from models.adcsp.flow_prior import AdCSPPFlowPrior
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
            feature_config=model_cfg.get("feature_schema", model_cfg.get("feature_config", {})),
        )

    def _adapter(self) -> AdCSPPTFRecordAdapter:
        model_cfg = self._config("model", {}) or {}
        mapping = dict(model_cfg.get("feature_mapping", {}) or {})
        schema = model_cfg.get("feature_schema", {}) or {}
        if schema.get("user_id_field"):
            mapping["user_id"] = schema["user_id_field"]
        if schema.get("labels"):
            mapping["labels"] = dict(schema["labels"])
        if schema.get("sequence_fields"):
            mapping["sequence"] = {name: name for name in schema["sequence_fields"]}
        return AdCSPPTFRecordAdapter(mapping, schema)

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
        max_train_examples = self.cfg.get("max_train_examples", None)
        if max_train_examples is not None:
            max_train_examples = int(max_train_examples)
            if max_train_examples <= 0:
                raise ValueError("max_train_examples must be positive when provided")
        log_every_batches = max(1, int(self.cfg.get("log_every_batches", 20)))
        log_every_examples = max(1, int(self.cfg.get("log_every_examples", 1000)))
        last_metrics = {}
        first_example = None

        examples = []
        parsed_batches = 0
        parsed_records = 0
        logging.info(
            "AdCSPP training: parsing TFRecord examples (max_train_examples=%s)",
            max_train_examples or "all",
        )
        for batch in self.train_dataset:
            parsed_batches += 1
            features, labels = batch[:2]
            current_batch_size = batch_size(features, labels)
            for index in range(current_batch_size):
                if max_train_examples is not None and parsed_records >= max_train_examples:
                    break
                example = adapter.example(
                    features,
                    labels,
                    index,
                    latent_dim=int(model_cfg.get("latent_dim", 16)),
                    feature_config=model_cfg.get("feature_schema", model_cfg.get("feature_config", {})),
                )
                if first_example is None:
                    first_example = example
                examples.append(example)
                parsed_records += 1
            if parsed_batches % log_every_batches == 0:
                logging.info(
                    "AdCSPP training: parsed %d batches / %d examples",
                    parsed_batches,
                    parsed_records,
                )
            if max_train_examples is not None and parsed_records >= max_train_examples:
                break
        if not examples:
            raise ValueError("AdCSPPFlowTrainTask received an empty train dataset")

        logging.info(
            "AdCSPP training: parsed %d examples; fitting traffic prior",
            len(examples),
        )

        grouped = {}
        for example in examples:
            grouped.setdefault(example.state.user_id, []).append(example)
        for user_examples in grouped.values():
            user_examples.sort(key=lambda item: item.context.event_timestamp)

        checkpoint_path = self.cfg.get(
            "checkpoint_path",
            model_cfg.get("checkpoint_path", "checkpoints/adcsp_model.weights.h5"),
        )
        prior_path = self.cfg.get(
            "flow_prior_path",
            model_cfg.get("flow_prior_path", str(Path(checkpoint_path).with_name("adcsp_flow_prior.json"))),
        )
        flow_prior = AdCSPPFlowPrior.fit(examples, seed=int(model_cfg.get("seed", 42)))
        flow_prior.save(prior_path)
        logging.info(
            "AdCSPP training: traffic prior saved to %s; starting %d behavior epoch(s)",
            prior_path,
            epochs,
        )

        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                click_probabilities, conversion_probabilities = [], []
                click_labels, conversion_labels = [], []
                # Carry a differentiable latent state through each user's
                # chronological path.  This makes the transition network
                # learn from future click/conversion supervision instead of
                # being a random post-processing component.
                processed_examples = 0
                for user_examples in grouped.values():
                    latent = tf.convert_to_tensor(user_examples[0].state.latent_state, dtype=tf.float32)
                    for example in user_examples:
                        click_probability, conversion_probability = model.differentiable_probabilities(
                            example.state,
                            example.context,
                            click_window,
                            conversion_window,
                            latent_state=latent,
                        )
                        click_probabilities.append(tf.reshape(click_probability, []))
                        conversion_probabilities.append(tf.reshape(conversion_probability, []))
                        click_labels.append(example.is_click)
                        conversion_labels.append(example.is_conversion)
                        latent = model.differentiable_transition(
                            latent,
                            example.state,
                            example.context,
                            example.is_click,
                            example.is_conversion,
                        )
                        processed_examples += 1
                        if processed_examples % log_every_examples == 0:
                            logging.info(
                                "AdCSPP training: epoch %d/%d processed %d/%d behavior examples",
                                epoch + 1,
                                epochs,
                                processed_examples,
                                len(examples),
                            )

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
            logging.info(
                "AdCSPP training: epoch %d/%d finished, %d behavior examples, loss=%.6f",
                epoch + 1,
                epochs,
                processed_examples,
                last_metrics["loss"],
            )

        # Build the transition network too.  Otherwise a checkpoint trained
        # only on row-wise behavior losses would silently restore without the
        # state-transition variables used by path generation.
        if first_example is not None:
            model.transition_state(first_example.state, first_example.context, 0, 0, first_example.context.event_timestamp)
        checkpoint = Path(checkpoint_path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        model.built = True
        model.save_weights(str(checkpoint))
        self.result = {"checkpoint_path": str(checkpoint), "flow_prior_path": str(prior_path), **last_metrics}
        logging.info("AdCSPP behavior training finished: %s", self.result)


class AdCSPPSyntheticTrainTask(AdCSPPFlowTrainTask):
    """Offline-only training task backed by the logical trajectory fixture."""

    step = "synthetic_train"
    self_managed = True
    uses_custom_model = True

    def run(self) -> None:
        from inputs.adcsp.synthetic_trajectory import SyntheticTrajectoryConfig, SyntheticTrajectoryGenerator

        synthetic = self.cfg.get("synthetic", {}) or {}
        config = SyntheticTrajectoryConfig(
            num_users=int(synthetic.get("num_users", 32)),
            exposures_per_user=int(synthetic.get("exposures_per_user", 8)),
            sequence_length=int(synthetic.get("sequence_length", 20)),
            seed=int(synthetic.get("seed", 7)),
            start_timestamp=int(synthetic.get("start_timestamp", 1_700_000_000)),
            exposure_gap_seconds=int(synthetic.get("exposure_gap_seconds", 300)),
        )
        self.train_dataset = SyntheticTrajectoryGenerator(config).as_dataset(
            batch_size=int(synthetic.get("batch_size", 32))
        )
        super().run()


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
                example = adapter.example(features, labels, index, latent_dim=int(model_cfg.get("latent_dim", 16)), feature_config=model_cfg.get("feature_schema", model_cfg.get("feature_config", {})))
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


class AdCSPPFlowGenerateTask(_AdCSPPTaskBase):
    """Generate continuous user paths from real platform request contexts.

    The platform still supplies TFRecord features.  This task does not create
    or serialize JSON and does not choose an ad: it rolls the trained behavior
    model over the request contexts already present in the input stream,
    samples click/conversion events, and feeds each event into the next state.
    """

    step = "generate"
    self_managed = False
    uses_custom_model = True

    @staticmethod
    def _jsonable(value):
        if hasattr(value, "numpy"):
            value = value.numpy()
        if isinstance(value, np.ndarray):
            return [AdCSPPFlowGenerateTask._jsonable(item) for item in value.tolist()]
        if isinstance(value, (np.generic,)):
            return value.item()
        if isinstance(value, dict):
            return {str(key): AdCSPPFlowGenerateTask._jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [AdCSPPFlowGenerateTask._jsonable(item) for item in value]
        return value

    @classmethod
    def _traffic_row(cls, step: dict) -> dict:
        context = step["context"]
        if isinstance(context, dict):
            context_features = dict(context.get("features", {}) or {})
            slot_id = context.get("slot_id", "unknown")
            event_timestamp = context.get("event_timestamp", step["timestamp"])
        else:
            context_features = dict(getattr(context, "features", {}) or {})
            slot_id = context.slot_id
            event_timestamp = context.event_timestamp
        state_after = step.get("state_after")
        row = {
            **context_features,
            "auid": step["user_id"],
            "slot_id": slot_id,
            "logid": context_features.get("logid", event_timestamp),
            "is_click": int(step["is_click"]),
            "is_conversion": int(step["is_conversion"]),
            "click_probability": float(step["click_probability"]),
            "conversion_probability": float(step["conversion_probability"]),
        }
        if state_after is not None:
            for name, values in state_after.sequence_features.items():
                row[name] = list(values)
        return cls._jsonable(row)

    def _save_generated_traffic(self, generated: list[dict]) -> str | None:
        output_cfg = self.cfg.get("output", {}) or {}
        if not bool(output_cfg.get("save_traffic", False)):
            return None
        output_dir = Path(output_cfg.get("output_dir", "outputs"))
        file_name = str(output_cfg.get("file_name", "generated_traffic.jsonl"))
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / file_name
        with output_path.open("w", encoding="utf-8") as handle:
            for step in generated:
                handle.write(json.dumps(self._traffic_row(step), ensure_ascii=False) + "\n")
        logging.info("AdCSPP generated traffic saved to %s (%d rows)", output_path, len(generated))
        return str(output_path)

    def _target_records(self) -> int | None:
        """Return the requested total row count, if generation is bounded."""
        value = self.cfg.get("num_records", self.cfg.get("target_records", self.cfg.get("n_gen")))
        if value is None:
            return None
        value = int(value)
        if value <= 0:
            raise ValueError("num_records must be positive when provided")
        return value

    def _configured_steps(self) -> int | None:
        """Read an optional fixed-length override for backward compatibility."""
        value = self.cfg.get("num_steps")
        return None if value is None else max(1, int(value))

    def run(self) -> None:
        dataset = getattr(self, "generate_dataset", None)
        if dataset is None:
            dataset = getattr(self, "test_dataset", None)
        prior_only = bool(self.cfg.get("generate_from_prior", False))
        if dataset is None and not prior_only:
            raise ValueError("AdCSPPFlowGenerateTask requires the platform generation/test dataset")

        model_cfg = self._config("model", {}) or {}
        model = self._model()
        adapter = self._adapter()
        click_window = int(self.cfg.get("click_window_seconds", 30))
        conversion_window = int(self.cfg.get("conversion_window_seconds", 3600))
        generator = AdCSPPPathGenerator(
            model,
            click_window_seconds=click_window,
            conversion_window_seconds=conversion_window,
            max_sequence_length=int(self.cfg.get("max_sequence_length", 20)),
            seed=int(self.cfg.get("seed", model_cfg.get("seed", 42))),
        )
        checkpoint_path = self.cfg.get(
            "checkpoint_path",
            model_cfg.get("checkpoint_path", "checkpoints/adcsp_model.weights.h5"),
        )
        prior_path = self.cfg.get(
            "flow_prior_path",
            model_cfg.get("flow_prior_path", str(Path(checkpoint_path).with_name("adcsp_flow_prior.json"))),
        )
        flow_prior = AdCSPPFlowPrior.load(prior_path) if bool(self.cfg.get("generate_new_contexts", False)) else None
        target_records = self._target_records()
        configured_steps = self._configured_steps()

        if dataset is None and prior_only:
            # Fully synthetic mode still requires the learned prior produced
            # by the training task.  There is deliberately no hand-written
            # feature fallback here.
            flow_prior = AdCSPPFlowPrior.load(prior_path)
            seed_profile = flow_prior.sample_profile()
            seed_state = UserStateBuilder.build(seed_profile, latent_dim=int(model_cfg.get("latent_dim", 16)))
            seed_context = flow_prior.sample_context(seed_profile, int(self.cfg.get("start_timestamp", 1)))
            model.differentiable_probabilities(seed_state, seed_context, click_window, conversion_window)
            model.transition_state(seed_state, seed_context, 0, 0, seed_context.event_timestamp)
            model.built = True
            self._restore(model, checkpoint_path)
            generated = []
            generated_users = 0
            requested_users = self.cfg.get("num_users")
            if target_records is None and requested_users is None:
                raise ValueError("prior-only generation requires num_records or num_users")
            while (target_records is not None and len(generated) < target_records) or (
                target_records is None and generated_users < int(requested_users)
            ):
                index = generated_users
                profile = flow_prior.sample_profile(user_id=f"generated_user_{index}")
                state = UserStateBuilder.build(profile, latent_dim=int(model_cfg.get("latent_dim", 16)))
                steps = generator.generate(
                    state=state,
                    prior=flow_prior,
                    num_steps=configured_steps,
                    start_timestamp=int(self.cfg.get("start_timestamp", 1)),
                    sample=bool(self.cfg.get("sample", True)),
                )
                generated.extend(step.to_dict() for step in steps)
                generated_users += 1
            if target_records is not None:
                generated = generated[:target_records]
            self.results = {"trajectories": generated}
            traffic_path = self._save_generated_traffic(generated)
            self.result = {"user_count": generated_users, "step_count": len(generated), "results": self.results, "traffic_path": traffic_path}
            return

        # Group the stream by user and use request timestamps to restore the
        # chronological path.  Labels, if present, are intentionally ignored:
        # generation samples outcomes from the trained model.
        grouped = {}
        for batch in dataset:
            features, labels = batch[:2]
            for index in range(batch_size(features, labels)):
                example = adapter.example(
                    features,
                    labels,
                    index,
                    latent_dim=int(model_cfg.get("latent_dim", 16)),
                    feature_config=model_cfg.get("feature_schema", model_cfg.get("feature_config", {})),
                )
                grouped.setdefault(example.state.user_id, []).append(example)

        generated = []
        restored = False
        for user_id, examples in grouped.items():
            examples.sort(key=lambda item: item.context.event_timestamp)
            initial_state = examples[0].state.clone()
            contexts = [item.context for item in examples]
            if not restored and contexts:
                model.differentiable_probabilities(initial_state, contexts[0], click_window, conversion_window)
                model.transition_state(initial_state, contexts[0], 0, 0, contexts[0].event_timestamp)
                model.built = True
                self._restore(model, self.cfg.get("checkpoint_path", model_cfg.get("checkpoint_path")))
                restored = True
            if flow_prior is not None:
                steps = generator.generate(
                    state=initial_state,
                    prior=flow_prior,
                    num_steps=configured_steps,
                    start_timestamp=int(contexts[-1].event_timestamp if contexts else initial_state.last_update_timestamp),
                    sample=bool(self.cfg.get("sample", True)),
                )
            else:
                steps = generator.rollout(
                    initial_state,
                    contexts,
                    sample=bool(self.cfg.get("sample", True)),
                )
            generated.extend(step.to_dict() for step in steps)
            if target_records is not None and len(generated) >= target_records:
                break

        if target_records is not None:
            generated = generated[:target_records]

        self.results = {"trajectories": generated}
        traffic_path = self._save_generated_traffic(generated)
        self.result = {
            "user_count": len(grouped),
            "step_count": len(generated),
            "results": self.results,
            "traffic_path": traffic_path,
        }
        logging.info(
            "AdCSPP continuous path generation finished: users=%d steps=%d",
            self.result["user_count"],
            self.result["step_count"],
        )

