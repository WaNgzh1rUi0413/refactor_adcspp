"""Field-complete synthetic traffic generator for local unit tests."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np

from models.adcsp.model import AdCSPPModel
from models.adcsp.simulator import ConditionalSimulator
from models.adcsp.state import UserStateBuilder
from inputs.adcsp.schema import ExposureContext, SimulationConfig, UserProfile


@dataclass(frozen=True)
class SyntheticTrafficConfig:
    num_users: int = 8
    exposures_per_user: int = 6
    output_dir: str = "traffic_output"
    seed: int = 42
    start_timestamp: int = 1_700_000_000
    horizon_seconds: int = 86_400
    click_window_seconds: int = 30
    conversion_window_seconds: int = 3_600
    latent_dim: int = 16
    feature_config: dict = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationResult:
    output_path: str
    user_count: int
    event_count: int


class SyntheticTrafficGenerator:
    def __init__(self, config: SyntheticTrafficConfig | None = None, model: AdCSPPModel | None = None):
        self.config = config or SyntheticTrafficConfig()
        self.model = model or AdCSPPModel(latent_dim=self.config.latent_dim, seed=self.config.seed, feature_config=self.config.feature_config)

    def _choices(self, field_name: str, fallback):
        values = list(self.config.feature_config.get("categorical_vocab", {}).get(field_name, fallback))
        # ``unknown`` is reserved for padding/OOV encoding and should not be
        # sampled as a real synthetic business value.
        if len(values) > 1:
            values = [value for value in values if str(value) != "unknown"]
        return values or list(fallback)

    def _range(self, field_name: str, fallback):
        values = self.config.feature_config.get("numeric_ranges", {}).get(field_name, fallback)
        return float(values[0]), float(values[1])

    def build_profile(self, user_index: int, rng: np.random.Generator) -> UserProfile:
        interest_pool = self._choices("ad_category", ["commerce", "content", "finance", "travel"])
        interest_count = min(2, len(interest_pool))
        interests = tuple(rng.choice(interest_pool, size=interest_count, replace=False).tolist()) if interest_count else tuple()
        return UserProfile(
            user_id=f"user_{user_index:05d}",
            gender=str(rng.choice(self._choices("gender", ["female", "male"]))),
            age_group=str(rng.choice(self._choices("age_group", ["18_24", "25_34", "35_44", "45_plus"]))),
            region=str(rng.choice(self._choices("region", ["north", "east", "south", "west"]))),
            device_type=str(rng.choice(self._choices("device_type", ["phone", "tablet", "desktop"]))),
            os_type=str(rng.choice(self._choices("os_type", ["android", "ios", "windows"]))),
            user_segment=str(rng.choice(self._choices("user_segment", ["new", "active", "high_intent"]))),
            user_long_term_interest=interests,
            user_registration_age=float(rng.uniform(*self._range("user_registration_age", [10, 2_000]))),
            user_value_segment=str(rng.choice(self._choices("user_value_segment", ["low", "medium", "high"]))),
        )

    def build_exposure_plan(self, profile: UserProfile, rng: np.random.Generator) -> List[ExposureContext]:
        current = self.config.start_timestamp + int(rng.integers(*map(int, self._range("initial_offset_seconds", [0, 300]))))
        contexts: List[ExposureContext] = []
        for index in range(self.config.exposures_per_user):
            current += int(rng.integers(*map(int, self._range("exposure_gap_seconds", [60, 1_800]))))
            category = str(rng.choice(self._choices("ad_category", ["commerce", "content", "finance", "travel"])))
            contexts.append(ExposureContext(
                impression_id=f"IMP_{profile.user_id}_{index:04d}",
                event_timestamp=current,
                slot_id=str(rng.choice(self._choices("slot_id", ["slot_a", "slot_b", "slot_c"]))),
                slot_type=str(rng.choice(self._choices("slot_type", ["feed", "search", "detail"]))),
                slot_position=str(rng.choice(self._choices("slot_position", ["top", "middle", "bottom"]))),
                ad_id=f"ad_{int(rng.integers(*map(int, self._range("ad_id", [1, 50])))):03d}",
                creative_id=f"creative_{int(rng.integers(*map(int, self._range("creative_id", [1, 30])))):03d}",
                campaign_id=f"campaign_{int(rng.integers(*map(int, self._range("campaign_id", [1, 10])))):02d}",
                advertiser_id=f"advertiser_{int(rng.integers(*map(int, self._range("advertiser_id", [1, 15])))):02d}",
                ad_category=category,
                objective_type=str(rng.choice(self._choices("objective_type", ["click", "activate", "purchase"]))),
                creative_type=str(rng.choice(self._choices("creative_type", ["image", "video", "native"]))),
                creative_size=str(rng.choice(self._choices("creative_size", ["square", "portrait", "landscape"]))),
                bid_price=float(rng.uniform(*self._range("bid_price", [0.2, 3.0]))),
                ecpm=float(rng.uniform(*self._range("ecpm", [1.0, 20.0]))),
                ad_quality_score=float(rng.uniform(*self._range("ad_quality_score", [0.4, 1.0]))),
                landing_page_type=str(rng.choice(self._choices("landing_page_type", ["product", "content", "form"]))),
                promotion_target=str(rng.choice(self._choices("promotion_target", ["purchase", "activate", "signup"]))),
                session_id=f"session_{profile.user_id}_{index // 3}",
                traffic_source=str(rng.choice(self._choices("traffic_source", ["organic", "search", "push"]))),
                app_category=str(rng.choice(self._choices("app_category", ["shopping"]))),
                page_context=str(rng.choice(self._choices("page_context", ["home", "search", "detail"]))),
                network_type=str(rng.choice(self._choices("network_type", ["wifi", "4g", "5g"]))),
                is_holiday=False,
            ))
        return contexts

    def generate(self) -> GenerationResult:
        rng = np.random.default_rng(self.config.seed)
        simulation_config = SimulationConfig(
            horizon_seconds=self.config.horizon_seconds,
            click_window_seconds=self.config.click_window_seconds,
            conversion_window_seconds=self.config.conversion_window_seconds,
            latent_dim=self.config.latent_dim,
            seed=self.config.seed,
            start_timestamp=self.config.start_timestamp,
            feature_config=self.config.feature_config,
        )
        simulator = ConditionalSimulator(self.model, simulation_config)
        records = []
        for user_index in range(self.config.num_users):
            profile = self.build_profile(user_index, rng)
            state = UserStateBuilder.build(profile, latent_dim=self.config.latent_dim)
            plan = self.build_exposure_plan(profile, rng)
            records.extend(simulator.simulate(state, plan).to_records())

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "traffic_event_ledger.jsonl"
        with output_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        return GenerationResult(str(output_path), self.config.num_users, len(records))
