"""Basic import tests to verify package structure."""

import pytest


def test_concordia_mini_imports():
    """Test that concordia_mini core modules import."""
    from concordia_mini.typing import entity_component
    from concordia_mini.language_model import language_model
    assert entity_component is not None
    assert language_model is not None


def test_negotiation_imports():
    """Test that negotiation modules import."""
    from negotiation import config, constants
    from negotiation.components import theory_of_mind
    assert config is not None
    assert constants is not None


def test_interpretability_scenarios():
    """Test that scenarios are defined."""
    from interpretability.emergent_prompts import EMERGENT_SCENARIOS
    assert len(EMERGENT_SCENARIOS) == 6
    assert "ultimatum_bluff" in EMERGENT_SCENARIOS
    assert "alliance_betrayal" in EMERGENT_SCENARIOS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
