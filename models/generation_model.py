import json
import logging
import math
import secrets
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm.auto import trange, tqdm


class MinMaxScaler:
    def __init__(self) -> None:
        self.min_: Optional[np.ndarray] = None
        self.max_: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> None:
        self.min_ = x.min(axis=0)
        self.max_ = x.max(axis=0)
        close = (self.max_ - self.min_) < 1e-6
        self.max_ = self.max_.copy()
        self.max_[close] = self.min_[close] + 1e-6

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.min_ is None or self.max_ is None:
            raise RuntimeError("MinMaxScaler must be fitted before transform")
        return (x - self.min_) / (self.max_ - self.min_)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        if self.min_ is None or self.max_ is None:
            raise RuntimeError("MinMaxScaler must be fitted before inverse")
        return x * (self.max_ - self.min_) + self.min_


class KerasMLP(tf.keras.layers.Layer):
    def __init__(self, din: int, dh: int, dout: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dense1 = tf.keras.layers.Dense(dh, activation="relu")
        self.dense2 = tf.keras.layers.Dense(dh, activation="relu")
        self.dense3 = tf.keras.layers.Dense(dout)

    def call(self, x: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        x = self.dense1(x, training=training)
        x = self.dense2(x, training=training)
        return self.dense3(x, training=training)


class TimeEmbedLayer(tf.keras.layers.Layer):
    def __init__(self, dim: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dim = dim

    def call(self, t: tf.Tensor) -> tf.Tensor:
        half = self.dim // 2
        half_f = tf.cast(half, tf.float32)
        freqs = tf.exp(-math.log(10000.0) * tf.range(half, dtype=tf.float32) / half_f)
        t_f = tf.cast(t, tf.float32)
        args = t_f[:, None] * freqs[None, :]
        emb = tf.concat([tf.sin(args), tf.cos(args)], axis=-1)
        if self.dim % 2 == 1:
            emb = tf.concat([emb, tf.zeros_like(emb[:, :1])], axis=-1)
        return emb


class CondDenoiserTF(tf.keras.layers.Layer):
    def __init__(self, zdim: int, tdim: int = 64, cdim: int = 32, hidden: int = 256, **kwargs) -> None:
        super().__init__(**kwargs)
        self.t_emb = TimeEmbedLayer(tdim)
        self.mlp = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(hidden, activation="relu"),
                tf.keras.layers.Dense(hidden, activation="relu"),
                tf.keras.layers.Dense(zdim),
            ]
        )
        self._zdim = zdim
        self._tdim = tdim
        self._cdim = cdim

    def call(self, zt: tf.Tensor, t: tf.Tensor, cond_vec: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        te = self.t_emb(t)
        h = tf.concat([zt, te, cond_vec], axis=-1)
        return self.mlp(h, training=training)


class VAETF(tf.keras.Model):
    def __init__(
        self,
        num_dim: int,
        zdim: int,
        hidden: int,
        cat_sizes: Dict[str, int],
        text_sizes: Dict[str, int],
        cat_emb_dims: Dict[str, int],
        text_emb_dims: Dict[str, int],
        cat_col_order: List[str],
        text_col_order: List[str],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.zd = zdim
        self.cat_col_order = list(cat_col_order)
        self.text_col_order = list(text_col_order)
        self.cat_embs = {
            c: tf.keras.layers.Embedding(cat_sizes[c], cat_emb_dims[c], name=f"emb_cat_{c}") for c in self.cat_col_order
        }
        self.text_embs = {
            c: tf.keras.layers.Embedding(text_sizes[c], text_emb_dims[c], name=f"emb_text_{c}")
            for c in self.text_col_order
        }
        enc_in = num_dim + sum(cat_emb_dims[c] for c in self.cat_col_order) + sum(text_emb_dims[c] for c in self.text_col_order)
        self.enc = KerasMLP(enc_in, hidden, 2 * zdim, name="vae_enc")
        self.dec = KerasMLP(zdim, hidden, num_dim, name="vae_dec")
        self.cat_heads = {c: tf.keras.layers.Dense(cat_sizes[c], name=f"head_cat_{c}") for c in self.cat_col_order}
        self.text_heads = {c: tf.keras.layers.Dense(text_sizes[c], name=f"head_text_{c}") for c in self.text_col_order}

    def encode(
        self,
        x_num: tf.Tensor,
        cat_idx: Dict[str, tf.Tensor],
        text_idx: Dict[str, tf.Tensor],
        training: Optional[bool] = None,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        parts = [x_num]
        for c in self.cat_col_order:
            parts.append(self.cat_embs[c](cat_idx[c], training=training))
        for c in self.text_col_order:
            parts.append(self.text_embs[c](text_idx[c], training=training))
        h = self.enc(tf.concat(parts, axis=-1), training=training)
        return h[:, : self.zd], h[:, self.zd :]

    @staticmethod
    def reparam(mu: tf.Tensor, lv: tf.Tensor) -> tf.Tensor:
        lv = tf.clip_by_value(lv, -30.0, 20.0)
        std = tf.exp(0.5 * lv)
        eps = tf.random.normal(tf.shape(std))
        return mu + eps * std

    def decode(
        self, z: tf.Tensor, training: Optional[bool] = None
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
        recon = self.dec(z, training=training)
        cat_logits = {c: self.cat_heads[c](z, training=training) for c in self.cat_col_order}
        text_logits = {c: self.text_heads[c](z, training=training) for c in self.text_col_order}
        return recon, cat_logits, text_logits


class GenerationModel(tf.keras.Model):
    """Tabular VAE + 条件 DDPM，基于 TensorFlow/Keras，便于 SavedModel 与 ONNX 导出。"""

    _BASE36 = "0123456789abcdefghijklmnopqrstuvwxyz"
    _NUM_CLIP_MIN = 0.0
    _NUM_CLIP_MAX = 50.0
    _DEFAULT_OAID_POOL_SIZE = 128

    def __init__(self, cfg: Any, **kwargs) -> None:
        super().__init__(**kwargs)
        # 避免 Keras 将 cfg 包装成 TrackedDict 导致 cfg.model 等属性访问失效
        object.__setattr__(self, "cfg", cfg)
        self.schema = list(cfg.schema)
        self.num_cols = dict(cfg.get("num_cols", {}))
        self.cat_cols_topk = dict(cfg.get("cat_cols_topk", {}))
        self.text_cols_topk = dict(cfg.get("text_cols_topk", {}))
        self.id_cols = ["slotid", "aid", "adid", "auid", "oaid"]
        self.cond_col = str(cfg.get("cond_col", "timehour"))

        self.seed = int(cfg.get("seed", 42))
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

        self._init_hparams()
        self._init_artifacts()

        self.vae: Optional[VAETF] = None
        self.den: Optional[CondDenoiserTF] = None
        self.emb_cond: Optional[tf.keras.layers.Embedding] = None
        self.onnx_export_model: Optional[tf.keras.Model] = None

        self.betas: Optional[tf.Tensor] = None
        self.alphas: Optional[tf.Tensor] = None
        self.abar: Optional[tf.Tensor] = None

        self.cat_col_order = list(self.cat_cols_topk.keys())
        self.text_col_order = list(self.text_cols_topk.keys())

    def _init_hparams(self) -> None:
        m = self.cfg.model
        self.z_dim = int(m.z_dim)
        self.hidden = int(m.hidden)
        self.vae_epochs = int(m.vae_epochs)
        self.vae_batch = int(m.vae_batch)
        self.vae_lr = float(m.vae_lr)
        self.kl_beta_start = float(m.kl_beta_start)
        self.kl_beta_end = float(m.kl_beta_end)
        self.timesteps = int(m.timesteps)
        self.ddpm_steps = int(m.ddpm_steps)
        self.denoiser_lr = float(m.denoiser_lr)
        self.cfg_drop_prob = float(m.cfg_drop_prob)
        self.cfg_scale = float(m.cfg_scale)
        self.slotid_len_fixed = int(m.slotid_len_fixed)
        self.aid_hex_len_fixed = int(m.aid_hex_len_fixed)
        self.emb_dim = int(m.get("emb_dim", 32))

    def _init_artifacts(self) -> None:
        self.vocabs: Dict[str, Dict[str, int]] = {}
        self.oov_idx: Dict[str, int] = {}
        self.vocab_items: Dict[str, List[str]] = {}
        self.vocab_prob: Dict[str, np.ndarray] = {}
        self.scaler = MinMaxScaler()

    @staticmethod
    def _artifact_paths(filepath: Path) -> Tuple[Path, Path]:
        p = Path(filepath)
        stem = p.stem
        parent = p.parent
        ckpt_dir = parent / f"{stem}_tf_ckpt"
        meta = parent / f"{stem}_tf.meta.json"
        return ckpt_dir, meta

    def save(self, filepath: str, overwrite: bool = True, include_optimizer: bool = False, **kwargs: Any) -> None:
        """保存两套产物：SavedModel(pb/variables/assets) + *_tf_ckpt/*_tf.meta.json。"""
        del include_optimizer  # 与 Keras save 接口对齐；当前不写入优化器状态
        if self.vae is None or self.den is None or self.emb_cond is None:
            raise RuntimeError("Model is not trained; call fit() before save()")

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        # 先导出 TF 原生 SavedModel 目录（saved_model.pb/variables/assets）。
        self.export_saved_model(str(path))
        ckpt_dir, mpath = self._artifact_paths(path)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        ckpt = tf.train.Checkpoint(vae=self.vae, den=self.den, emb_cond=self.emb_cond)
        manager = tf.train.CheckpointManager(ckpt, str(ckpt_dir), max_to_keep=2)
        manager.save()
        logging.info("Saved generation checkpoint to %s", ckpt_dir)

        def _json_default(obj: Any):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj) if isinstance(obj, np.floating) else int(obj)
            raise TypeError(f"Not JSON serializable: {type(obj)}")

        meta = {
            "num_keys": list(self.num_keys),
            "has_real_num": bool(self._has_real_num),
            "vocabs": {k: {str(kk): int(vi) for kk, vi in v.items()} for k, v in self.vocabs.items()},
            "oov_idx": {k: int(v) for k, v in self.oov_idx.items()},
            "vocab_items": {k: [str(x) for x in v] for k, v in self.vocab_items.items()},
            "vocab_prob": {k: np.asarray(v, dtype=np.float64).tolist() for k, v in self.vocab_prob.items()},
            "scaler_min": None if self.scaler.min_ is None else np.asarray(self.scaler.min_, dtype=np.float64).tolist(),
            "scaler_max": None if self.scaler.max_ is None else np.asarray(self.scaler.max_, dtype=np.float64).tolist(),
            "oaid_pool": [str(x) for x in self.oaid_pool],
            "oaid_prob": np.asarray(self.oaid_prob, dtype=np.float64).tolist(),
            "schema": [str(x) for x in self.schema],
            "num_cols": {str(k): dict(v) if isinstance(v, dict) else v for k, v in self.num_cols.items()},
            "cat_cols_topk": {k: int(v) for k, v in self.cat_cols_topk.items()},
            "text_cols_topk": {k: int(v) for k, v in self.text_cols_topk.items()},
            "cond_col": str(self.cond_col),
        }
        with open(mpath, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, default=_json_default)
        logging.info("Saved generation model meta to %s", mpath)

    def export_saved_model(self, export_dir: str) -> None:
        """导出可被 mcp 使用的 SavedModel（包含 saved_model.pb）。"""
        if self.onnx_export_model is None:
            raise RuntimeError("Model is not ready for SavedModel export; call fit() first")
        target = Path(export_dir)
        target.mkdir(parents=True, exist_ok=True)
        tf.saved_model.save(self.onnx_export_model, str(target))
        logging.info("Saved SavedModel (pb) to %s", target)

    def load_weights(self, filepath: str, **kwargs: Any) -> None:
        """从 save() 产出的路径加载：可与 checkpoint 基名（如 .pt）对应。"""
        del kwargs
        ckpt_dir, mpath = self._artifact_paths(Path(filepath))
        latest = tf.train.latest_checkpoint(str(ckpt_dir))
        if not latest:
            raise ValueError(f"No TensorFlow checkpoint found under {ckpt_dir}")
        if self.vae is None or self.den is None or self.emb_cond is None:
            raise RuntimeError("load_weights requires networks built (run fit() first in this process)")
        ckpt = tf.train.Checkpoint(vae=self.vae, den=self.den, emb_cond=self.emb_cond)
        ckpt.restore(latest).expect_partial()
        if mpath.exists():
            with open(mpath, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self._restore_meta(meta)

    def _restore_meta(self, meta: Dict[str, Any]) -> None:
        self.num_keys = list(meta["num_keys"])
        self._has_real_num = bool(meta["has_real_num"])
        self.vocabs = {k: {str(kk): int(vi) for kk, vi in v.items()} for k, v in meta["vocabs"].items()}
        self.oov_idx = {k: int(v) for k, v in meta["oov_idx"].items()}
        self.vocab_items = {k: list(v) for k, v in meta["vocab_items"].items()}
        self.vocab_prob = {k: np.asarray(v, dtype=np.float64) for k, v in meta["vocab_prob"].items()}
        smin, smax = meta["scaler_min"], meta["scaler_max"]
        self.scaler.min_ = None if smin is None else np.asarray(smin, dtype=np.float64)
        self.scaler.max_ = None if smax is None else np.asarray(smax, dtype=np.float64)
        self.oaid_pool = list(meta["oaid_pool"])
        self.oaid_prob = np.asarray(meta["oaid_prob"], dtype=np.float64)
        self.schema = list(meta["schema"])
        self.num_cols = dict(meta["num_cols"])
        self.cat_cols_topk = {k: int(v) for k, v in meta["cat_cols_topk"].items()}
        self.text_cols_topk = {k: int(v) for k, v in meta["text_cols_topk"].items()}
        self.cond_col = str(meta["cond_col"])
        self.cat_col_order = list(self.cat_cols_topk.keys())
        self.text_col_order = list(self.text_cols_topk.keys())

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """默认未训练时占位；训练结束后推理走 generate()。"""
        del inputs, training
        raise NotImplementedError("Use generate() for sampling; use onnx_export_model for ONNX.")

    @staticmethod
    def _build_vocab(series: pd.Series, topk: int) -> Tuple[Dict[str, int], int, List[str], np.ndarray]:
        vc = series.fillna("").astype(str).value_counts()
        items = vc.index.tolist()[:topk]
        vocab = {s: i for i, s in enumerate(items)}
        oov = len(items)
        probs = vc.loc[items].values.astype(np.float64) + 1e-8
        probs = probs / probs.sum()
        return vocab, oov, items, probs

    def _cat_to_idx(self, col: str, value: Any) -> int:
        if pd.isna(value):
            value = ""
        return self.vocabs[col].get(str(value), self.oov_idx[col])

    @staticmethod
    def _emb_dim(topk: int) -> int:
        return min(32, max(2, int(math.log2(max(topk, 2)))))

    def _schedule_tf(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        betas = tf.linspace(1e-4, 2e-2, self.timesteps)
        alphas = 1.0 - betas
        abar = tf.math.cumprod(alphas)
        return betas, alphas, abar

    @classmethod
    def _sanitize_numeric_series(cls, series: pd.Series) -> pd.Series:
        series = pd.to_numeric(series, errors="coerce")
        series = series.replace([np.inf, -np.inf], np.nan).fillna(0)
        return series.clip(cls._NUM_CLIP_MIN, cls._NUM_CLIP_MAX)

    def _build_default_oaid_pool(self) -> Tuple[List[str], np.ndarray]:
        pool = [f"oaid-{i:04d}" for i in range(1, self._DEFAULT_OAID_POOL_SIZE + 1)]
        prob = np.ones(len(pool), dtype=np.float64) / len(pool)
        return pool, prob

    def _sample_vocab_values(
        self,
        col: str,
        logits: tf.Tensor,
        temperature: Optional[float] = None,
    ) -> List[str]:
        logits_np = logits.numpy()
        if temperature is None:
            idx = np.argmax(logits_np, axis=-1)
        else:
            p = tf.nn.softmax(logits_np / temperature).numpy()
            idx = np.array([np.random.choice(np.arange(p.shape[1]), p=p[i]) for i in range(p.shape[0])])

        items = self.vocab_items[col]
        vals: List[str] = []
        for t in idx:
            if t >= len(items):
                vals.append(str(np.random.choice(items, p=self.vocab_prob[col])))
            else:
                vals.append(items[int(t)])
        return vals

    def _build_onnx_export_model(self, cond_vocab_size: int) -> None:
        zt_in = tf.keras.Input(shape=(self.z_dim,), dtype=tf.float32, name="zt")
        t_in = tf.keras.Input(shape=(), dtype=tf.int64, name="t")
        c_in = tf.keras.Input(shape=(), dtype=tf.int32, name="cond_ids")
        c_vec = self.emb_cond(c_in)
        out = self.den(zt_in, t_in, c_vec)
        self.onnx_export_model = tf.keras.Model(
            inputs=[zt_in, t_in, c_in],
            outputs=out,
            name="generation_denoiser_step",
        )

    def fit(self, df_raw: pd.DataFrame, df_valid: Optional[pd.DataFrame] = None) -> None:
        object.__setattr__(self, "df_raw", df_raw.copy())

        if self.cond_col not in self.df_raw.columns:
            default_cond = str(self.cfg.get("target_timehour", 0))
            self.df_raw[self.cond_col] = default_cond
        if self.cond_col not in self.cat_cols_topk:
            self.cat_cols_topk[self.cond_col] = int(self.cfg.get("cond_topk", 16))

        for col, k in self.cat_cols_topk.items():
            vocab, oov, items, p = self._build_vocab(self.df_raw[col], k)
            self.vocabs[col], self.oov_idx[col], self.vocab_items[col], self.vocab_prob[col] = vocab, oov, items, p

        for col, k in self.text_cols_topk.items():
            vocab, oov, items, p = self._build_vocab(self.df_raw[col], k)
            self.vocabs[col], self.oov_idx[col], self.vocab_items[col], self.vocab_prob[col] = vocab, oov, items, p

        self.cat_col_order = list(self.cat_cols_topk.keys())
        self.text_col_order = list(self.text_cols_topk.keys())

        num_keys = list(self.num_cols.keys())
        self._has_real_num = len(num_keys) > 0
        if not self._has_real_num:
            num_keys = ["__dummy_num__"]
        x_num = np.zeros((len(df_raw), len(num_keys)), dtype=np.float32)
        if self._has_real_num:
            for j, col in enumerate(num_keys):
                x = self._sanitize_numeric_series(self.df_raw[col])
                x_num[:, j] = x.astype(np.float32)

        self.scaler.fit(x_num)
        x_scaled = self.scaler.transform(x_num).astype(np.float32)

        if len(x_scaled) == 0:
            raise ValueError("No source rows after preprocessing; cannot train generation model")

        y_cat = {
            c: self.df_raw[c].apply(lambda v: self._cat_to_idx(c, v)).astype(np.int64).values for c in self.cat_cols_topk
        }
        y_text = {
            c: self.df_raw[c].apply(lambda v: self._cat_to_idx(c, v)).astype(np.int64).values for c in self.text_cols_topk
        }

        use_external_valid = df_valid is not None and len(df_valid) > 0
        tr_idx = np.arange(len(x_scaled))

        xtr = x_scaled[tr_idx].astype(np.float32)
        ycat_tr = {c: y_cat[c][tr_idx].astype(np.int64) for c in self.cat_cols_topk}
        ytxt_tr = {c: y_text[c][tr_idx].astype(np.int64) for c in self.text_cols_topk}

        if use_external_valid:
            df_va = df_valid.copy()
            if self.cond_col not in df_va.columns:
                df_va[self.cond_col] = str(self.cfg.get("target_timehour", 0))

            for col in self.cat_cols_topk:
                if col not in df_va.columns:
                    df_va[col] = ""
            for col in self.text_cols_topk:
                if col not in df_va.columns:
                    df_va[col] = ""

            x_num_va = np.zeros((len(df_va), len(num_keys)), dtype=np.float32)
            if self._has_real_num:
                for j, col in enumerate(num_keys):
                    if col in df_va.columns:
                        x = self._sanitize_numeric_series(df_va[col])
                    else:
                        x = pd.Series(np.zeros(len(df_va), dtype=np.float32))
                    x_num_va[:, j] = x.astype(np.float32)

            xva_scaled = self.scaler.transform(x_num_va).astype(np.float32)
            y_cat_va = {c: df_va[c].apply(lambda v: self._cat_to_idx(c, v)).astype(np.int64).values for c in self.cat_cols_topk}
            y_text_va = {c: df_va[c].apply(lambda v: self._cat_to_idx(c, v)).astype(np.int64).values for c in self.text_cols_topk}

            xva = xva_scaled.astype(np.float32)
            ycat_va = {c: y_cat_va[c] for c in self.cat_cols_topk}
            ytxt_va = {c: y_text_va[c] for c in self.text_cols_topk}
        else:
            rng = np.random.default_rng(self.seed)
            perm = rng.permutation(len(x_scaled))
            train_ratio = float(self.cfg.get("train_ratio", 0.9))
            if not 0.0 <= train_ratio <= 1.0:
                raise ValueError(f"train_ratio must be in [0, 1], got {train_ratio}")

            cut = int(train_ratio * len(perm))
            if len(perm) == 1:
                cut = 1
            else:
                cut = min(max(cut, 1), len(perm) - 1)
            tr_idx, va_idx = perm[:cut], perm[cut:]

            xtr = x_scaled[tr_idx].astype(np.float32)
            ycat_tr = {c: y_cat[c][tr_idx].astype(np.int64) for c in self.cat_cols_topk}
            ytxt_tr = {c: y_text[c][tr_idx].astype(np.int64) for c in self.text_cols_topk}
            xva = x_scaled[va_idx].astype(np.float32)
            ycat_va = {c: y_cat[c][va_idx].astype(np.int64) for c in self.cat_cols_topk}
            ytxt_va = {c: y_text[c][va_idx].astype(np.int64) for c in self.text_cols_topk}

        logging.info(
            "VAE dataset rows train=%d valid=%d (external_valid=%s)",
            len(xtr),
            len(xva),
            use_external_valid,
        )

        cat_sizes = {c: len(self.vocab_items[c]) + 1 for c in self.cat_cols_topk}
        text_sizes = {c: len(self.vocab_items[c]) + 1 for c in self.text_cols_topk}
        cat_emb_dims = {c: self._emb_dim(self.cat_cols_topk[c]) for c in self.cat_cols_topk}
        text_emb_dims = {c: self._emb_dim(self.text_cols_topk[c]) for c in self.text_cols_topk}

        self.vae = VAETF(
            num_dim=len(num_keys),
            zdim=self.z_dim,
            hidden=self.hidden,
            cat_sizes=cat_sizes,
            text_sizes=text_sizes,
            cat_emb_dims=cat_emb_dims,
            text_emb_dims=text_emb_dims,
            cat_col_order=self.cat_col_order,
            text_col_order=self.text_col_order,
            name="vae",
        )

        opt_vae = tf.keras.optimizers.Adam(learning_rate=self.vae_lr)

        def vae_losses(
            xb: tf.Tensor, ycat_b: Dict[str, tf.Tensor], ytxt_b: Dict[str, tf.Tensor], beta: tf.Tensor
        ) -> tf.Tensor:
            mu, lv = self.vae.encode(xb, ycat_b, ytxt_b, training=True)
            lv_safe = tf.clip_by_value(lv, -30.0, 20.0)
            z = VAETF.reparam(mu, lv_safe)
            recon, cat_logits, text_logits = self.vae.decode(z, training=True)

            mse = tf.reduce_mean(tf.square(recon - xb))
            ce_cat = 0.0
            for c in self.cat_col_order:
                ce_cat = ce_cat + tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ycat_b[c], logits=cat_logits[c])
                )
            ce_txt = 0.0
            for c in self.text_col_order:
                ce_txt = ce_txt + tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ytxt_b[c], logits=text_logits[c])
                )
            kld = -0.5 * tf.reduce_mean(1.0 + lv_safe - tf.square(mu) - tf.exp(lv_safe))
            return mse + 0.4 * ce_cat + 0.2 * ce_txt + beta * kld

        n_train = len(xtr)
        n_val = len(xva)

        for ep in trange(self.vae_epochs, desc="[VAE]"):
            beta = self.kl_beta_start + (self.kl_beta_end - self.kl_beta_start) * ep / max(1, self.vae_epochs - 1)
            beta_t = tf.constant(beta, dtype=tf.float32)
            perm = np.random.permutation(n_train) if n_train > 0 else np.array([], dtype=np.int64)
            for start in range(0, n_train, self.vae_batch):
                idx = perm[start : start + self.vae_batch]
                if len(idx) == 0:
                    continue
                if len(idx) < self.vae_batch and n_train >= self.vae_batch:
                    continue
                xb = tf.constant(xtr[idx], dtype=tf.float32)
                ycat_b = {c: tf.constant(ycat_tr[c][idx], dtype=tf.int64) for c in self.cat_col_order}
                ytxt_b = {c: tf.constant(ytxt_tr[c][idx], dtype=tf.int64) for c in self.text_col_order}

                with tf.GradientTape() as tape:
                    loss = vae_losses(xb, ycat_b, ytxt_b, beta_t)
                if not tf.math.is_finite(loss):
                    tqdm.write(f"[warn] ep{ep + 1:02d} skipped non-finite train loss")
                    continue
                vars_ = self.vae.trainable_variables
                grads = tape.gradient(loss, vars_)
                grads, _ = tf.clip_by_global_norm(grads, 5.0)
                opt_vae.apply_gradients(zip(grads, vars_))

            val_losses: List[float] = []
            if n_val > 0:
                for start in range(0, n_val, self.vae_batch):
                    idx = np.arange(start, min(start + self.vae_batch, n_val))
                    xb = tf.constant(xva[idx], dtype=tf.float32)
                    ycat_b = {c: tf.constant(ycat_va[c][idx], dtype=tf.int64) for c in self.cat_col_order}
                    ytxt_b = {c: tf.constant(ytxt_va[c][idx], dtype=tf.int64) for c in self.text_col_order}
                    loss_item = float(vae_losses(xb, ycat_b, ytxt_b, beta_t).numpy())
                    if np.isfinite(loss_item):
                        val_losses.append(loss_item)
                    else:
                        tqdm.write(f"[warn] ep{ep + 1:02d} ignored non-finite valid loss")
            if val_losses:
                tqdm.write(f"ep{ep + 1:02d} valid={np.mean(val_losses):.4f}")
            else:
                tqdm.write(f"ep{ep + 1:02d} valid=n/a (empty or non-finite validation split)")

        z_rows: List[tf.Tensor] = []
        for start in range(0, n_train, self.vae_batch):
            idx = np.arange(start, min(start + self.vae_batch, n_train))
            xb = tf.constant(xtr[idx], dtype=tf.float32)
            ycat_b = {c: tf.constant(ycat_tr[c][idx], dtype=tf.int64) for c in self.cat_col_order}
            ytxt_b = {c: tf.constant(ytxt_tr[c][idx], dtype=tf.int64) for c in self.text_col_order}
            mu, _ = self.vae.encode(xb, ycat_b, ytxt_b, training=False)
            z_rows.append(mu)
        if not z_rows:
            raise ValueError("No latent rows generated from training dataset")
        ztr = tf.concat(z_rows, axis=0)

        cond_train = ycat_tr[self.cond_col]

        self.emb_cond = tf.keras.layers.Embedding(cat_sizes[self.cond_col], self.emb_dim, name="emb_cond")
        self.den = CondDenoiserTF(self.z_dim, cdim=self.emb_dim, hidden=self.hidden, name="denoiser")
        self.betas, self.alphas, self.abar = self._schedule_tf()

        opt_den = tf.keras.optimizers.Adam(learning_rate=self.denoiser_lr)
        opt_emb = tf.keras.optimizers.Adam(learning_rate=self.denoiser_lr * 0.5)

        ztr_np = ztr.numpy()
        n_z = ztr_np.shape[0]

        for step in trange(self.ddpm_steps, desc="[LDM]"):
            bs = min(256, n_z)
            sel = np.random.randint(0, n_z, size=bs)
            z0 = tf.constant(ztr_np[sel], dtype=tf.float32)
            t = tf.random.uniform((bs,), maxval=self.timesteps, dtype=tf.int32)
            a_bar = tf.gather(self.abar, t)[:, None]
            eps = tf.random.normal(tf.shape(z0))
            zt = tf.sqrt(a_bar) * z0 + tf.sqrt(1.0 - a_bar) * eps
            cond_ids = tf.constant(cond_train[sel], dtype=tf.int32)
            cond_vec = self.emb_cond(cond_ids)
            drop = tf.cast(tf.random.uniform((bs, 1)) < self.cfg_drop_prob, tf.float32)
            cond_in = cond_vec * (1.0 - drop)

            t64 = tf.cast(t, tf.int64)
            with tf.GradientTape() as tape:
                pred = self.den(zt, t64, cond_in, training=True)
                loss = tf.reduce_mean(tf.square(pred - eps))
            vars_den = self.den.trainable_variables
            vars_emb = self.emb_cond.trainable_variables
            grads = tape.gradient(loss, vars_den + vars_emb)
            grads_den = grads[: len(vars_den)]
            grads_emb = grads[len(vars_den) :]
            pairs_den = [(g, v) for g, v in zip(grads_den, vars_den) if g is not None]
            pairs_emb = [(g, v) for g, v in zip(grads_emb, vars_emb) if g is not None]
            if pairs_den:
                gs, vs = zip(*pairs_den)
                gs_clipped, _ = tf.clip_by_global_norm(list(gs), 5.0)
                opt_den.apply_gradients(zip(gs_clipped, vs))
            if pairs_emb:
                gs, vs = zip(*pairs_emb)
                gs_clipped, _ = tf.clip_by_global_norm(list(gs), 5.0)
                opt_emb.apply_gradients(zip(gs_clipped, vs))

            if (step + 1) % 200 == 0:
                tqdm.write(f"step {step + 1} loss={float(loss.numpy()):.5f}")

        default_pool, default_prob = self._build_default_oaid_pool()
        if "oaid" in self.df_raw.columns:
            oaid_counts = self.df_raw["oaid"].fillna("").astype(str).value_counts()
            oaid_counts = oaid_counts[oaid_counts.index != ""]
            if len(oaid_counts) > 0:
                self.oaid_pool = oaid_counts.index.tolist()
                self.oaid_prob = (oaid_counts.values.astype(np.float64) / oaid_counts.values.sum())
            else:
                self.oaid_pool = default_pool
                self.oaid_prob = default_prob
        else:
            self.oaid_pool = default_pool
            self.oaid_prob = default_prob

        self.num_keys = num_keys
        self._build_onnx_export_model(cat_sizes[self.cond_col])

    def _p_sample(self, n: int, cond_labels: Dict[str, tf.Tensor], cfg_scale: float) -> tf.Tensor:
        assert self.den is not None and self.emb_cond is not None and self.betas is not None and self.alphas is not None
        z = tf.random.normal((n, self.z_dim))
        c = self.emb_cond(cond_labels[self.cond_col])
        c0 = tf.zeros_like(c)

        for t in trange(self.timesteps - 1, -1, -1, desc="[Sampling]"):
            tt = tf.fill((n,), t)
            tt64 = tf.cast(tt, tf.int64)
            eps_c = self.den(z, tt64, c, training=False)
            eps_uc = self.den(z, tt64, c0, training=False)
            eps = eps_uc + cfg_scale * (eps_c - eps_uc)
            a = self.alphas[t]
            z = (z - tf.sqrt(1.0 - a) * eps) / tf.sqrt(a)
            if t > 0:
                z = z + tf.sqrt(self.betas[t]) * tf.random.normal(tf.shape(z))
        return z

    def _decode_typed(self, z: tf.Tensor) -> pd.DataFrame:
        assert self.vae is not None
        recon_num, cat_logits, text_logits = self.vae.decode(z, training=False)

        x_np = recon_num.numpy().astype(np.float32)
        x_np = self.scaler.inverse(x_np)

        out: Dict[str, Any] = {}
        if self._has_real_num:
            for j, col in enumerate(self.num_keys):
                out[col] = np.clip(np.round(x_np[:, j]), self._NUM_CLIP_MIN, self._NUM_CLIP_MAX).astype(int)

        for col in self.cat_cols_topk:
            out[col] = self._sample_vocab_values(col, cat_logits[col], temperature=1.3)

        for col in self.text_cols_topk:
            out[col] = self._sample_vocab_values(col, text_logits[col])

        n = int(z.shape[0])
        row_oaids = np.random.choice(self.oaid_pool, size=n, p=self.oaid_prob)
        cache: Dict[str, Tuple[str, str, str]] = {}
        aids, adids, auids = [], [], []
        for oid in row_oaids:
            if oid not in cache:
                cache[oid] = (
                    self._gen_hex(self.aid_hex_len_fixed),
                    self._gen_uuid(),
                    self._gen_uuid(),
                )
            a, d, u = cache[oid]
            aids.append(a)
            adids.append(d)
            auids.append(u)

        out["slotid"] = [self._gen_base36(self.slotid_len_fixed) for _ in range(n)]
        out["aid"] = aids
        out["adid"] = adids
        out["auid"] = auids
        out["oaid"] = list(row_oaids)

        df = pd.DataFrame(out)
        for col in self.schema:
            if col not in df.columns:
                df[col] = ""
        return df[self.schema]

    def _gen_base36(self, n_chars: int) -> str:
        return "".join(secrets.choice(self._BASE36) for _ in range(n_chars))

    @staticmethod
    def _gen_uuid() -> str:
        return str(uuid.uuid4())

    @staticmethod
    def _gen_hex(n_chars: int) -> str:
        n_bytes = (n_chars + 1) // 2
        return secrets.token_hex(n_bytes)[:n_chars]

    def generate(self, n: int, target_timehour: int, cfg_scale: Optional[float] = None) -> pd.DataFrame:
        if cfg_scale is None:
            cfg_scale = self.cfg_scale
        cond_idx = self.vocabs[self.cond_col].get(str(target_timehour), self.oov_idx[self.cond_col])
        cond = {self.cond_col: tf.fill((n,), cond_idx)}
        z = self._p_sample(n, cond, cfg_scale)
        return self._decode_typed(z)

    @staticmethod
    def _ks_stat(x: np.ndarray, y: np.ndarray) -> float:
        x = np.asarray(x)
        y = np.asarray(y)
        x = x[np.isfinite(x)]
        y = y[np.isfinite(y)]
        if len(x) == 0 or len(y) == 0:
            return 0.0
        xs = np.sort(x)
        ys = np.sort(y)
        z = np.sort(np.unique(np.concatenate([xs, ys])))

        def ecdf(a, t):
            return np.searchsorted(a, t, side="right") / len(a)

        d = 0.0
        for t in z[:: max(1, len(z) // 2000)]:
            d = max(d, abs(ecdf(xs, t) - ecdf(ys, t)))
        return float(d)

    @staticmethod
    def _tv_dist(p: Dict[str, float], q: Dict[str, float]) -> float:
        keys = set(p.keys()) | set(q.keys())
        s = 0.0
        for k in keys:
            s += abs(p.get(k, 0.0) - q.get(k, 0.0))
        return 0.5 * s

    @staticmethod
    def _js_div(p, q, eps: float = 1e-12):
        keys = set(p.keys()) | set(q.keys())
        if not keys:
            return 0.0

        p_vec = np.array([p.get(k, 0.0) for k in keys], dtype=np.float64)
        q_vec = np.array([q.get(k, 0.0) for k in keys], dtype=np.float64)

        p_sum = p_vec.sum()
        q_sum = q_vec.sum()
        if p_sum <= 0 or q_sum <= 0:
            return 0.0

        p_vec = p_vec / p_sum
        q_vec = q_vec / q_sum
        m_vec = 0.5 * (p_vec + q_vec)

        p_vec = np.clip(p_vec, eps, 1.0)
        q_vec = np.clip(q_vec, eps, 1.0)
        m_vec = np.clip(m_vec, eps, 1.0)

        kl_pm = np.sum(p_vec * np.log(p_vec / m_vec))
        kl_qm = np.sum(q_vec * np.log(q_vec / m_vec))
        return float(0.5 * (kl_pm + kl_qm))

    @staticmethod
    def _freq_dist(series: pd.Series, topk: int = 50) -> Dict[str, float]:
        vc = series.fillna("").astype(str).value_counts(normalize=True)
        return vc.head(topk).to_dict()

    def validate(self, real_df: pd.DataFrame, gen_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        ks_rows = []
        for col in self.num_cols.keys():
            xr = pd.to_numeric(real_df[col], errors="coerce").values
            xg = pd.to_numeric(gen_df[col], errors="coerce").values
            ks_rows.append({"feature": col, "KS_D": self._ks_stat(xr, xg)})

        tv_rows = []
        for col in list(self.cat_cols_topk.keys()) + list(self.text_cols_topk.keys()):
            pr = self._freq_dist(real_df[col], topk=100)
            pg = self._freq_dist(gen_df[col], topk=100)
            tv_rows.append(
                {
                    "feature": col,
                    "TV_top100": self._tv_dist(pr, pg),
                    "JS_top100": self._js_div(pr, pg),
                }
            )

        ks_df = pd.DataFrame(ks_rows, columns=["feature", "KS_D"])
        if not ks_df.empty:
            ks_df = ks_df.sort_values("KS_D", ascending=False)

        tv_df = pd.DataFrame(tv_rows, columns=["feature", "TV_top100", "JS_top100"])
        if not tv_df.empty:
            tv_df = tv_df.sort_values("TV_top100", ascending=False)
        return ks_df, tv_df
