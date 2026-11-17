"""Build model state from online static and precomputed historical features."""

from __future__ import annotations

from typing import Mapping, Optional

import numpy as np

from inputs.adcsp.schema import UserProfile, UserState


class UserStateBuilder:
    @staticmethod
    def build(
        profile: UserProfile,
        aggregate_features: Optional[Mapping[str, object]] = None,
        latent_dim: int = 16,
        static_embedding: Optional[np.ndarray] = None,
        feature_config: Optional[Mapping[str, object]] = None,
    ) -> UserState:
        del feature_config  # Sequence semantics are supplied by the upstream input pipeline.
        sequence_features = {
            str(key): list(value)
            for key, value in (aggregate_features or {}).items()
            if str(key).startswith("hist_") and isinstance(value, (list, tuple, np.ndarray))
        }
        dynamic_features = {
            str(key): value
            for key, value in (aggregate_features or {}).items()
            if not str(key).startswith("hist_")
        }
        return UserState(
            user_id=profile.user_id,
            static_features=profile.to_dict(),
            static_embedding=np.zeros(8, dtype=np.float32)
            if static_embedding is None
            else np.asarray(static_embedding, dtype=np.float32).copy(),
            latent_state=np.zeros(latent_dim, dtype=np.float32),
            last_update_timestamp=0,
            dynamic_features=dynamic_features,
            sequence_features={key: list(value) for key, value in sequence_features.items()},
            initial_sequence_features={key: list(value) for key, value in sequence_features.items()},
        )

