"""Field-complete synthetic traffic generator for local unit tests."""

from __future__ import annotations

import json
from dataclasses import dataclass
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


@dataclass(frozen=True)
class GenerationResult:
    output_path: str
    user_count: int
    event_count: int


class SyntheticTrafficGenerator:
    def __init__(self, config: SyntheticTrafficConfig | None = None, model: AdCSPPModel | None = None):
        self.config = config or SyntheticTrafficConfig()
        self.model = model or AdCSPPModel(latent_dim=self.config.latent_dim, seed=self.config.seed)

    def build_profile(self, user_index: int, rng: np.random.Generator) -> UserProfile:
        return UserProfile(
            user_id=f"user_{user_index:05d}",
            gender=str(rng.choice(["female", "male"])),
            age_group=str(rng.choice(["18_24", "25_34", "35_44", "45_plus"])),
            region=str(rng.choice(["north", "east", "south", "west"])),
            device_type=str(rng.choice(["phone", "tablet", "desktop"])),
            os_type=str(rng.choice(["android", "ios", "windows"])),
            user_segment=str(rng.choice(["new", "active", "high_intent"])),
            user_long_term_interest=tuple(rng.choice(["commerce", "content", "finance", "travel"], size=2, replace=False).tolist()),
            user_registration_age=float(rng.integers(10, 2_000)),
            user_value_segment=str(rng.choice(["low", "medium", "high"])),
        )

    def build_exposure_plan(self, profile: UserProfile, rng: np.random.Generator) -> List[ExposureContext]:
        current = self.config.start_timestamp + int(rng.integers(0, 300))
        contexts: List[ExposureContext] = []
        for index in range(self.config.exposures_per_user):
            current += int(rng.integers(60, 1_800))
            category = str(rng.choice(["commerce", "content", "finance", "travel"]))
            contexts.append(ExposureContext(
                impression_id=f"IMP_{profile.user_id}_{index:04d}",
                event_timestamp=current,
                slot_id=str(rng.choice(["slot_a", "slot_b", "slot_c"])),
                slot_type=str(rng.choice(["feed", "search", "detail"])),
                slot_position=str(rng.choice(["top", "middle", "bottom"])),
                ad_id=f"ad_{int(rng.integers(1, 50)):03d}",
                creative_id=f"creative_{int(rng.integers(1, 30)):03d}",
                campaign_id=f"campaign_{int(rng.integers(1, 10)):02d}",
                advertiser_id=f"advertiser_{int(rng.integers(1, 15)):02d}",
                ad_category=category,
                objective_type=str(rng.choice(["click", "activate", "purchase"])),
                creative_type=str(rng.choice(["image", "video", "native"])),
                creative_size=str(rng.choice(["square", "portrait", "landscape"])),
                bid_price=float(rng.uniform(0.2, 3.0)),
                ecpm=float(rng.uniform(1.0, 20.0)),
                ad_quality_score=float(rng.uniform(0.4, 1.0)),
                landing_page_type=str(rng.choice(["product", "content", "form"])),
                promotion_target=str(rng.choice(["purchase", "activate", "signup"])),
                session_id=f"session_{profile.user_id}_{index // 3}",
                traffic_source=str(rng.choice(["organic", "search", "push"])),
                app_category="shopping",
                page_context=str(rng.choice(["home", "search", "detail"])),
                network_type=str(rng.choice(["wifi", "4g", "5g"])),
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
