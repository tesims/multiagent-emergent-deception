"""Core experiment configuration for mechanistic interpretability research.

This is the central configuration hub for all deception detection experiments.
Modify these dataclasses to customize your experiments.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Literal


@dataclass
class ModelConfig:
    """LLM and interpretability model configuration."""

    # Model selection
    name: str = "google/gemma-2-2b-it"
    """HuggingFace model name. Tested models:
    - google/gemma-2-2b-it (fast, ~4GB VRAM)
    - google/gemma-2-9b-it (better, ~20GB VRAM)
    - meta-llama/Llama-3.1-8B-Instruct
    """

    # Hardware
    device: Literal["cuda", "cpu", "mps"] = "cuda"
    dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16"

    # TransformerLens settings
    use_transformerlens: bool = True
    """Enable activation capture via TransformerLens."""

    cache_activations: bool = True
    """Cache activations for probe training."""

    # SAE settings
    use_sae: bool = True
    """Enable Sparse Autoencoder analysis."""

    sae_release: str = "gemma-scope-2b-pt-res-canonical"
    """SAE Lens release name."""

    sae_layer: int = 20
    """Which layer to analyze with SAE (0-indexed)."""

    sae_id: str = "layer_20/width_16k/canonical"
    """Specific SAE ID within the release."""


@dataclass
class EvaluatorConfig:
    """Lightweight evaluator model for ground truth extraction.

    The evaluator is a separate (usually smaller) model that judges whether
    the subject model's responses are deceptive. This separation ensures:
    1. The subject model is tested without self-evaluation bias
    2. Faster evaluation with a lightweight model
    3. Consistent evaluation across different subject models
    """

    enabled: bool = True
    """Enable separate evaluator model."""

    api: Literal["local", "together", "google", None] = "local"
    """Evaluator API:
    - 'local': Load google/gemma-2-2b-it locally (~2GB VRAM, no API needed)
    - 'together': Use Together AI API (requires TOGETHER_API_KEY)
    - 'google': Use Google AI Studio API (requires GOOGLE_API_KEY)
    - None: Use subject model for self-evaluation (not recommended)
    """

    local_model: str = "google/gemma-2-2b-it"
    """Model to use for local evaluation."""

    together_model: str = "google/gemma-2-4b-it"
    """Model to use with Together AI."""

    google_model: str = "gemini-1.5-flash"
    """Model to use with Google AI Studio."""

    max_tokens: int = 64
    """Max tokens for evaluator responses (short = faster)."""


@dataclass
class ProbeConfig:
    """Linear probe training configuration."""

    # Training
    train_ratio: float = 0.8
    """Train/test split ratio."""

    regularization: float = 1.0
    """L2 regularization strength."""

    max_iter: int = 1000
    """Maximum training iterations."""

    # Layers to probe
    layers_to_probe: List[int] = field(default_factory=lambda: [10, 15, 20, 25])
    """Which layers to train probes on."""

    # Validation
    min_accuracy: float = 0.6
    """Minimum accuracy threshold for valid probe."""

    run_sanity_checks: bool = True
    """Run probe sanity checks (shuffled labels, random features)."""


@dataclass
class CausalConfig:
    """Causal validation configuration (activation patching, ablation)."""

    enabled: bool = True
    """Enable causal validation."""

    num_samples: int = 30
    """Number of samples for causal interventions."""

    # Intervention types
    run_activation_patching: bool = True
    """Patch activations between deceptive/honest examples."""

    run_ablation: bool = True
    """Zero-ablate identified deception features."""

    run_steering: bool = True
    """Test steering vectors for behavior modification."""

    # Thresholds
    min_effect_size: float = 0.1
    """Minimum effect size to consider significant."""


@dataclass
class ScenarioConfig:
    """Deception scenario configuration."""

    mode: Literal["emergent", "instructed", "contest"] = "emergent"
    """Scenario mode:
    - emergent: Incentive-based deception (no explicit instructions)
    - instructed: Explicit deception instructions (Apollo-style)
    - contest: Game-theoretic scenarios (Fishery, Treaty, Gameshow)
    """

    scenarios: List[str] = field(default_factory=lambda: [
        "ultimatum_bluff",
        "capability_bluff",
        "hidden_value",
        "info_withholding",
        "promise_break",
        "alliance_betrayal",
    ])
    """Which scenarios to run. Available emergent scenarios:
    - ultimatum_bluff: Bluffing about walking away
    - capability_bluff: Overstating capabilities
    - hidden_value: Hiding true preferences
    - info_withholding: Strategic information hiding
    - promise_break: Making promises with intent to break
    - alliance_betrayal: Forming alliances to betray
    """

    num_trials: int = 50
    """Number of trials per scenario."""

    max_rounds: int = 3
    """Maximum conversation rounds per trial."""


@dataclass
class ExperimentConfig:
    """Main experiment configuration combining all settings."""

    # Sub-configs
    model: ModelConfig = field(default_factory=ModelConfig)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    probes: ProbeConfig = field(default_factory=ProbeConfig)
    causal: CausalConfig = field(default_factory=CausalConfig)
    scenarios: ScenarioConfig = field(default_factory=ScenarioConfig)

    # Output
    output_dir: str = "./results"
    """Directory for experiment outputs."""

    checkpoint_dir: str = "./checkpoints"
    """Directory for checkpoints (resume interrupted experiments)."""

    save_activations: bool = False
    """Save raw activations (large files)."""

    # Experiment metadata
    experiment_name: Optional[str] = None
    """Optional experiment name for organizing results."""

    random_seed: int = 42
    """Random seed for reproducibility."""

    # Logging
    verbose: bool = True
    log_to_file: bool = True


# Default configurations for common use cases
QUICK_TEST = ExperimentConfig(
    model=ModelConfig(name="google/gemma-2-2b-it"),
    scenarios=ScenarioConfig(scenarios=["alliance_betrayal"], num_trials=1),
    causal=CausalConfig(num_samples=10),
)

FULL_EXPERIMENT = ExperimentConfig(
    model=ModelConfig(name="google/gemma-2-9b-it"),
    scenarios=ScenarioConfig(num_trials=50),
    causal=CausalConfig(num_samples=30),
)

FAST_ITERATION = ExperimentConfig(
    model=ModelConfig(name="google/gemma-2-2b-it", use_sae=False),
    scenarios=ScenarioConfig(scenarios=["ultimatum_bluff"], num_trials=5),
    causal=CausalConfig(enabled=False),
)
