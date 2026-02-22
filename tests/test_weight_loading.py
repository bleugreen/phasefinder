"""Test that Phasefinder model weights load without errors."""

import pytest


def test_phasefinder_init():
    """Verify Phasefinder initializes and loads weights without RuntimeError."""
    from phasefinder import Phasefinder

    pf = Phasefinder(quiet=True)
    assert pf.device is not None
    assert pf.model is not None


def test_constants_importable():
    """Verify constants module is importable and has expected values."""
    from phasefinder.constants import FRAME_RATE, HOP, SAMPLE_RATE

    assert SAMPLE_RATE == 22050
    assert HOP == 512
    assert abs(FRAME_RATE - 43.066) < 0.01
