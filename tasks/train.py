import json
import logging
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from core.callbacks import LogLossCallback
from models.abstract_model import AbstractModel
from utils import global_variables as gv
from utils.config import constants
from utils.easydict import EasyDict
from utils.logger import log_decorator
from .abstract_task import AbstractTask
from core.model_export import export_onnx



class Train(AbstractTask):
    step = "train"

    def __init__(
        self,
        cfg: EasyDict = None,
        model: AbstractModel = None,
        train_dataset: tf.data.Dataset = None,
        valid_dataset: tf.data.Dataset = None,
    ):
        """
        Build a training task.

        :param cfg: training config
        :param model: trainable model
        :param train_dataset: training dataset
        :param valid_dataset: validation dataset
        """
        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        logging.info("Initializing task with config %s", cfg)
        self.cfg = cfg
        self.history = None
        self.checkpoint_path = None
        self.model_export_path = None

    def _check_input_arguments(self):
        """
        Validate whether model and dataset satisfy training requirements.
        """
        if not isinstance(self.train_dataset, tf.data.Dataset):
            raise ValueError(
                "Input train_dataset must be a 'tf.data.Dataset'. Got {}.".format(type(self.train_dataset))
            )
        if not isinstance(self.model, AbstractModel):
            raise ValueError("Input model must be an 'AbstractModel'. Got {}.".format(type(self.model)))
        if not getattr(self.model, "is_trainable", True):
            raise ValueError("Input model must be trainable")

    def _build_callbacks(self) -> list:
        """
        Build keras callbacks.
        """
        log_step = self.cfg.get("log_step", 10)
        callbacks = [
            LogLossCallback(self.cfg.get("log_step", log_step)),
        ]

        if self.cfg.get("is_use_r2", False):
            from metrics.callback_funcs.r2 import R2Validator
            r2_val_output = self.cfg.get("r2_val_output", [])
            r2_val_label = self.cfg.get("r2_val_label", [])
            callbacks.append(R2Validator(self.valid_dataset, r2_val_output, r2_val_label))

        save_checkpoint = self.cfg.get("save_checkpoint", True)
        save_best = self.cfg.get("save_best", False)
        monitor_metric = self.cfg.get("monitor_metric", "loss")
        monitor_mode = self.cfg.get("monitor_mode", "min")
        save_freq = self.cfg.get("save_freq", "epoch")
        if save_checkpoint:
            callbacks.append(
                keras.callbacks.ModelCheckpoint(
                    gv.PATH_CONFIG.checkpoint_path,
                    monitor=monitor_metric,
                    mode=monitor_mode,
                    verbose=1,
                    save_best_only=save_best,
                    save_freq=save_freq,
                    save_weights_only=True,
                )
            )

        if self.cfg.get("tensorboard", {}):
            tensorboard_callback = TensorBoard(
                log_dir=self.cfg.tensorboard.get("log_dir", constants.TENSORBOARD_LOG_DIR),
                histogram_freq=self.cfg.tensorboard.get("histogram_freq", 0),
                write_graph=self.cfg.tensorboard.get("write_graph", True),
                write_images=self.cfg.tensorboard.get("write_images", False),
                write_steps_per_second=self.cfg.tensorboard.get("write_steps_per_second", False),
                update_freq=self.cfg.tensorboard.get("update_freq", 10),
                profile_batch=self.cfg.tensorboard.get("profile_batch", 0),
                embeddings_freq=self.cfg.tensorboard.get("embeddings_freq", 0),
                embeddings_metadata=self.cfg.tensorboard.get("embeddings_metadata", None),
            )
            callbacks.append(tensorboard_callback)

        return callbacks

    def _train_by_keras_fit(self) -> None:
        epochs = self.cfg.epochs
        if isinstance(self.train_dataset, tf.data.Dataset):
            logging.info("Using dataset for training")
            batch_size = None
        else:
            batch_size = self.cfg.get("batch_size")

        self.history = self.model.fit(
            x=self.train_dataset,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=self.valid_dataset,
            callbacks=self._build_callbacks(),
            verbose=self.cfg.get("verbose", "auto"),
        )

    def _build_model_path(self) -> None:
        self.checkpoint_path = gv.PATH_CONFIG.checkpoint_path
        self.model_export_path = gv.PATH_CONFIG.model_export_path
        logging.info(
            "Checkpoints will go to %s. Exported model will be saved at %s",
            self.checkpoint_path,
            self.model_export_path,
        )

    @log_decorator
    def run_training(self):
        """
        Run training with model interface.
        """
        use_custom_train_loop = self.cfg.get("use_custom_train_loop", False)
        if use_custom_train_loop:
            self._train_loop()
        else:
            self._train_by_keras_fit()

    @log_decorator
    def export_model(self, path: str = None):
        """
        Export model.

        :param path: target export path. If None, use configured default path.
        """
        save_path = path or self.model_export_path

        logging.info("Try exporting model to %s", save_path)
        # If configured to export best model, load checkpoint weights first.
        if self.cfg.get("save_best", False):
            self.model.load_weights(self.checkpoint_path)
        self.model.save(save_path, include_optimizer=False)

        if self.cfg.get("save_model_train_best_score", False):
            save_train_path = self.cfg.get("save_model_train_best_train_score_path", "")
            save_valid_path = self.cfg.get("save_model_train_best_valid_score_path", "")
            if save_train_path:
                self._do_save_score(save_train_path, self.train_dataset)
            if save_valid_path:
                self._do_save_score(save_valid_path, self.valid_dataset)

        if self.cfg.get("is_use_model_version", False):
            self._save_model_version_file()

        if self.cfg.get("save_onnx", False):
            try:
                remove_dummy_opset = self.cfg.get("remove_dummy_opset", True)
                export_onnx(self.model, save_path, remove_dummy_opset)
            except Exception as e:
                logging.error(str(e))
        logging.info("Export model done")

    def _do_save_score(self, save_path, dataset):
        logging.info("do save score, save path:%s", save_path)
        try:
            date = gv.MTP_TASK_CONFIG.date
            save_path = save_path.replace("#period_date_placeholder", date)
            check_path = "/".join(save_path.split("/")[:-1])
            if check_path and (not os.path.exists(check_path)):
                os.makedirs(check_path)

            model_out = self.model.predict(dataset)
            if isinstance(model_out, dict):
                score_data = {}
                for k, v in model_out.items():
                    arr = np.array(v)
                    if arr.ndim > 1:
                        arr = arr.reshape(arr.shape[0], -1)[:, 0]
                    score_data[k] = arr
                score_df = pd.DataFrame(score_data)
            else:
                arr = np.array(model_out)
                if arr.ndim > 1:
                    arr = arr.reshape(arr.shape[0], -1)[:, 0]
                score_df = pd.DataFrame({"score": arr})

            score_df.to_csv(save_path, index=False)
        except Exception as e:
            logging.error(str(e))
        logging.info("do save score end")

    def _save_model_version_file(self):
        logging.info("start to save model version file.")
        logging.info("checkpoint path is %s", self.checkpoint_path)
        logging.info("model export path is %s", self.model_export_path)
        try:
            save_model_version_path = gv.PATH_CONFIG.model_version_path
            if not os.path.exists(save_model_version_path):
                os.makedirs(save_model_version_path)
            version_file = os.path.join(save_model_version_path, "model_version.csv")
            version_data = {}
            if os.path.exists(version_file):
                with open(version_file, "r", encoding="utf-8") as file:
                    json_str = file.readline().strip()
                    if json_str:
                        version_data = json.loads(json_str)
            version_data[gv.MTP_TASK_CONFIG.date] = self.checkpoint_path

            with open(version_file, "w", encoding="utf-8", newline="") as file:
                logging.info("model version file is %s", version_data)
                file.write(json.dumps(version_data, ensure_ascii=False))
            logging.info("end to save model version file.")

        except Exception as e:
            logging.error("save model version file error info: %s", e)

    @log_decorator
    def _train_loop(self):
        epochs = self.cfg.epochs
        batch_size = self.cfg.batch_size
        log_step = self.cfg.get("log_step", 10)

        for epoch in range(epochs):
            logging.info("\nStart of epoch %d", epoch)

            for step, (x_batch_train, y_batch_train) in enumerate(self.train_dataset):
                with tf.GradientTape() as tape:
                    logits = self.model(x_batch_train, training=True)
                    loss_value = self.model.loss_op(y_batch_train, logits)

                grads = tape.gradient(loss_value, self.model.trainable_weights)
                self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

                if step % log_step == 0:
                    logging.info("Training loss (for one batch) at step %d: %.4f", step, float(loss_value))
                    logging.info("Seen so far: %s samples" % ((step + 1) * batch_size))

    def run(self) -> None:
        if not self.model:
            raise ValueError("Model is not built for train task")
        if not self.model.is_trainable:
            self.model.build_trainable_model()
        self._check_input_arguments()

        self._build_model_path()
        self.run_training()
        self.model.summary()
        self.export_model()
