"""Central configuration for mechanistic interpretability experiments.

This module provides all configuration classes for running deception detection
experiments on LLM agents. Configure experiments by modifying these settings.

Quick Start:
    from config import ExperimentConfig, ModelConfig, ProbeConfig

    # Configure your experiment
    experiment = ExperimentConfig(
        model=ModelConfig(name="google/gemma-2-2b-it", device="cuda"),
        scenarios=["ultimatum_bluff", "alliance_betrayal"],
        num_trials=50,
    )

Available Configurations:
    - ExperimentConfig: Main experiment settings
    - ModelConfig: LLM and interpretability model settings
    - ProbeConfig: Linear probe training settings
    - CausalConfig: Activation patching/ablation settings
    - ScenarioConfig: Deception scenario settings
    - AgentConfig: Agent behavior settings (negotiation, future types)
"""

from config.experiment import (
    ExperimentConfig,
    ModelConfig,
    ProbeConfig,
    CausalConfig,
    ScenarioConfig,
)

from config.agents.negotiation import (
    StrategyConfig,
    ModuleDefaults,
    TheoryOfMindConfig,
    DeceptionDetectionConfig,
)

__all__ = [
    # Core experiment configs
    "ExperimentConfig",
    "ModelConfig",
    "ProbeConfig",
    "CausalConfig",
    "ScenarioConfig",
    # Agent configs
    "StrategyConfig",
    "ModuleDefaults",
    "TheoryOfMindConfig",
    "DeceptionDetectionConfig",
]
