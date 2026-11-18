"""Learned traffic feature prior used by complete path generation.

The prior is fitted from the real TFRecord examples that train AdCSPP.  It is
not a hard-coded vocabulary or a rule-based traffic simulator: static profiles,
request contexts, numeric values, and inter-exposure gaps are sampled from
empirical distributions observed in training data.  The behavior model then
adds learned click/conversion outcomes and the state machine feeds them back
into the next request.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from inputs.adcsp.schema import ExposureContext, UserProfile


def _plain(value):
    if hasattr(value, "numpy"):
        value = value.numpy()
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    return value


class AdCSPPFlowPrior:
    """Serializable empirical distribution learned from training examples."""

    def __init__(self, seed: int = 42):
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.profiles = []
        self.contexts = []
        self.contexts_by_segment = defaultdict(list)
        self.contexts_by_prev_slot = defaultdict(list)
        self.gaps = []
        self.path_lengths = []

    @classmethod
    def fit(cls, examples: Iterable, seed: int = 42) -> "AdCSPPFlowPrior":
        prior = cls(seed)
        users = {}
        timestamps = defaultdict(list)
        all_timestamps = []
        user_examples = defaultdict(list)
        for example in examples:
            profile = _plain(dict(example.state.static_features))
            profile_features = dict(profile.get("features", {}) or {})
            profile_features.update(_plain(example.state.dynamic_features))
            profile_features.update(_plain(example.state.sequence_features))
            profile["features"] = profile_features
            user_id = str(example.state.user_id)
            users.setdefault(user_id, profile)
            context = {
                **_plain(example.context.features),
                **_plain(example.context.to_dict()),
            }
            context.pop("features", None)
            context.pop("impression_id", None)
            context.pop("event_timestamp", None)
            prior.contexts.append(context)
            timestamps[user_id].append(int(example.context.event_timestamp))
            all_timestamps.append(int(example.context.event_timestamp))
            user_examples[user_id].append((int(example.context.event_timestamp), context))
        prior.profiles = list(users.values())
        for values in user_examples.values():
            values = [item[1] for item in sorted(values, key=lambda item: item[0])]
            prior.path_lengths.append(max(1, len(values)))
            for previous, current in zip(values, values[1:]):
                prior.contexts_by_prev_slot[str(previous.get("slot_id", "unknown"))].append(current)
        for values in timestamps.values():
            values = sorted(values)
            prior.gaps.extend(max(1, b - a) for a, b in zip(values, values[1:]))
        if not prior.gaps:
            # A batch containing one row per user has no within-user gap.  In
            # that case learn a global inter-arrival distribution from the
            # observed request timestamps rather than inventing a constant.
            values = sorted(all_timestamps)
            prior.gaps.extend(max(1, b - a) for a, b in zip(values, values[1:]))
        if not prior.profiles or not prior.contexts:
            raise ValueError("AdCSPPFlowPrior requires at least one real training example")
        return prior

    def _sample(self, values):
        if not values:
            raise ValueError("Cannot sample from an empty learned traffic prior")
        return values[int(self.rng.integers(0, len(values)))]

    def sample_profile(self, user_id: str | None = None) -> UserProfile:
        values = self._sample(self.profiles)
        return UserProfile(
            user_id=str(user_id or f"generated_user_{int(self.rng.integers(0, 2**31))}"),
            features=dict(values.get("features", values) or {}),
        )

    def sample_gap(self) -> int:
        # A missing gap can only happen for a one-row training user.  Use the
        # median observed gap, never an arbitrary business-rule timestamp.
        if not self.gaps:
            raise ValueError("The learned traffic prior has no inter-exposure gaps")
        return int(max(1, round(float(self._sample(self.gaps)))))

    def sample_path_length(self) -> int:
        """Sample a user trajectory length learned from real paths."""
        if not self.path_lengths:
            raise ValueError("The learned traffic prior has no user path lengths")
        return int(max(1, round(float(self._sample(self.path_lengths)))))

    def sample_context(self, profile: UserProfile, timestamp: int, previous_context=None) -> ExposureContext:
        if previous_context is not None:
            candidates = self.contexts_by_prev_slot.get(previous_context.slot_id, [])
        else:
            candidates = []
        candidates = candidates or self.contexts
        values = dict(self._sample(candidates))
        return ExposureContext(
            impression_id=f"generated_{profile.user_id}_{int(timestamp)}",
            event_timestamp=int(timestamp),
            slot_id=str(values.get("slot_id", "unknown")),
            features=dict(values),
        )

    def save(self, path: str | Path) -> None:
        payload = {
            "seed": self.seed, "profiles": self.profiles, "contexts": self.contexts,
            "contexts_by_segment": dict(self.contexts_by_segment),
            "contexts_by_prev_slot": dict(self.contexts_by_prev_slot), "gaps": self.gaps,
            "path_lengths": self.path_lengths,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "AdCSPPFlowPrior":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Learned traffic prior not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        prior = cls(int(payload.get("seed", 42)))
        prior.profiles = payload.get("profiles", [])
        prior.contexts = payload.get("contexts", [])
        prior.contexts_by_segment.update(payload.get("contexts_by_segment", {}))
        prior.contexts_by_prev_slot.update(payload.get("contexts_by_prev_slot", {}))
        prior.gaps = payload.get("gaps", [])
        prior.path_lengths = payload.get("path_lengths", [])
        if not prior.path_lengths:
            # Backward-compatible loading for priors created before trajectory
            # lengths were persisted.
            prior.path_lengths = [1]
        if not prior.profiles or not prior.contexts:
            raise ValueError(f"Learned traffic prior is empty: {path}")
        return prior

