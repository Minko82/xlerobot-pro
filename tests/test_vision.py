"""Run the vision/IK offline suite under pytest when its dependencies exist."""

import pytest


def test_vision_offline_suite():
    pytest.importorskip(
        "pinocchio", reason="vision extras not installed (requirements/requirements-vision.txt)"
    )
    pytest.importorskip("pink", reason="vision extras not installed (requirements/requirements-vision.txt)")

    from tests import vision_offline_suite

    assert vision_offline_suite.main() == 0
