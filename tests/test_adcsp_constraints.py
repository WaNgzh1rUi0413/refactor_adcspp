import unittest

from inputs.adcsp.schema import Event, EventType, UserProfile
from models.adcsp.constraints import EventValidationError, validate_ledger
from models.adcsp.state import StateUpdater, UserStateBuilder


class AdCSPPConstraintTest(unittest.TestCase):
    def test_click_without_exposure_is_rejected(self):
        profile = UserProfile("u1", "female", "25_34", "east", "phone", "ios", "active", ("commerce",), 10.0, "medium")
        state = UserStateBuilder.build(profile)
        with self.assertRaises(EventValidationError):
            StateUpdater().apply_event(state, Event("u1", "c1", 100, EventType.CLICK, impression_id="missing", click_id="c1"))

    def test_ledger_validator_detects_orphan_conversion(self):
        event = Event("u1", "v1", 100, EventType.CONVERSION, click_id="missing")
        errors = validate_ledger([event])
        self.assertTrue(any(error.startswith("orphan_conversion") for error in errors))

