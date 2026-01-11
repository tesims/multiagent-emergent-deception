# Multi-Agent Emergent Deception

**Mechanistic interpretability framework for detecting emergent deception in LLM agents.**

A research tool for studying how deception emerges in multi-agent LLM systems and detecting it through activation analysis.

## Core Purpose

This framework enables researchers to:
1. **Run deception scenarios** - Test LLM agents in situations that incentivize deception
2. **Capture activations** - Record internal model states during deceptive behavior
3. **Train detection probes** - Build classifiers that identify deception from activations
4. **Validate causally** - Confirm probes detect real deception circuits via interventions

## Quick Start

### Installation

```bash
git clone https://github.com/tesims/multiagent-emergent-deception.git
cd multiagent-emergent-deception
pip install -e .
```

### Run Experiment (CLI)

```bash
# Quick test
deception run --model google/gemma-2-2b-it --trials 5

# Full experiment with causal validation
deception run --model google/gemma-2-9b-it --trials 40 --causal

# List available scenarios
deception scenarios
```

### Run Experiment (Python)

```python
from config import ExperimentConfig
from interpretability import InterpretabilityRunner

# Auto-configure everything based on model
config = ExperimentConfig.for_model("google/gemma-2-2b-it", num_trials=10)
config.print_config_summary()

# Run experiment
runner = InterpretabilityRunner(
    model_name=config.model.name,
    device="cuda",
)

results = runner.run_all_emergent_scenarios(
    scenarios=config.scenarios.scenarios,
    trials_per_scenario=config.scenarios.num_trials,
)

# Save and analyze
runner.save_dataset("activations.pt")
```

### Google Colab

```python
# Install from GitHub
!pip install git+https://github.com/tesims/multiagent-emergent-deception.git

# Run experiment (2B model fits in free tier)
from config import ExperimentConfig
config = ExperimentConfig.for_model("google/gemma-2-2b-it", num_trials=5)
```

## Supported Models

Models auto-configure SAE settings and probe layers:

| Model | VRAM | SAE | Use Case |
|-------|------|-----|----------|
| `google/gemma-2-2b-it` | ~4GB | Yes | Fast iteration, Colab free |
| `google/gemma-2-9b-it` | ~20GB | Yes | Research quality |
| `google/gemma-2-27b-it` | ~54GB | Yes | Best performance |
| `meta-llama/Llama-3.1-8B-Instruct` | ~16GB | No | Alternative architecture |

```python
# Just change the model - everything else auto-configures
config = ExperimentConfig.for_model("google/gemma-2-9b-it")
```

## Configuration

All experiments are configured through `config/`:

```python
from config import ExperimentConfig, MODEL_PRESETS

# Option 1: Auto-configure (recommended)
config = ExperimentConfig.for_model("google/gemma-2-9b-it", num_trials=50)

# Option 2: Manual configuration
from config import ModelConfig, ScenarioConfig, ProbeConfig

config = ExperimentConfig(
    model=ModelConfig(name="google/gemma-2-9b-it"),
    probes=ProbeConfig.for_model("google/gemma-2-9b-it"),
    scenarios=ScenarioConfig(
        scenarios=["ultimatum_bluff", "alliance_betrayal"],
        num_trials=50,
    ),
)

# Option 3: Use presets
from config import QUICK_TEST, FULL_EXPERIMENT, FAST_ITERATION
config = QUICK_TEST  # 1 scenario, 1 trial
```

### Config Reference

| Config | Purpose |
|--------|---------|
| `ExperimentConfig` | Main experiment settings |
| `ModelConfig` | LLM, TransformerLens, SAE settings |
| `ProbeConfig` | Linear probe training |
| `CausalConfig` | Activation patching, ablation, steering |
| `ScenarioConfig` | Deception scenarios and trials |
| `StrategyConfig` | Agent negotiation behavior |
| `DeceptionDetectionConfig` | Linguistic deception cues |

## Deception Scenarios

### Emergent (Incentive-Based)
No explicit deception instructions - agents deceive because it's strategically advantageous:

| Scenario | Description |
|----------|-------------|
| `ultimatum_bluff` | Bluffing about walking away from negotiation |
| `capability_bluff` | Overstating capabilities or resources |
| `hidden_value` | Hiding true preferences to gain advantage |
| `info_withholding` | Strategically withholding information |
| `promise_break` | Making promises with intent to break |
| `alliance_betrayal` | Forming alliances only to betray |

### Instructed (Apollo-Style)
Explicit instructions to deceive - for baseline comparisons.

## Project Structure

```
multiagent-emergent-deception/
├── config/                    # CONFIGURATION
│   ├── experiment.py          # ExperimentConfig, ModelConfig, etc.
│   └── agents/
│       └── negotiation.py     # Agent behavior constants
│
├── interpretability/          # DECEPTION DETECTION
│   ├── cli.py                 # Click CLI (deception command)
│   ├── evaluation.py          # InterpretabilityRunner
│   ├── core/                  # DatasetBuilder, GroundTruthDetector
│   ├── scenarios/             # Deception scenarios
│   ├── probes/                # Probe training & analysis
│   └── causal/                # Causal validation
│
├── negotiation/               # AGENT IMPLEMENTATION
│   ├── components/            # Cognitive modules
│   └── game_master/           # GM components
│
├── concordia_mini/            # Framework dependency (Apache-2.0)
├── docs/                      # Documentation
│   ├── SETUP.md               # Detailed setup guide
│   ├── METHODOLOGY.md         # Technical methodology
│   ├── OUTPUT_GUIDE.md        # Output interpretation guide
│   ├── ARCHITECTURE.md        # System architecture diagrams
│   ├── RUNPOD_GUIDE.md        # Cloud GPU deployment
│   └── CONTRIBUTING.md        # Contribution guidelines
└── tests/                     # Test suite
```

## Documentation

| Document | Description |
|----------|-------------|
| **[docs/SETUP.md](docs/SETUP.md)** | Installation, configuration, Colab usage |
| **[docs/METHODOLOGY.md](docs/METHODOLOGY.md)** | Complete technical methodology |
| **[docs/OUTPUT_GUIDE.md](docs/OUTPUT_GUIDE.md)** | How to interpret experiment outputs |
| **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** | System architecture with diagrams |
| **[docs/RUNPOD_GUIDE.md](docs/RUNPOD_GUIDE.md)** | Cloud GPU deployment (RunPod) |
| **[docs/CONTRIBUTING.md](docs/CONTRIBUTING.md)** | Contribution guidelines |

## Citation

```bibtex
@software{sims2025deception,
  author = {Sims, Teanna},
  title = {Mechanistic Interpretability for Deception Detection in LLM Agents},
  year = {2025},
  url = {https://github.com/tesims/multiagent-emergent-deception}
}
```

## License

- `negotiation/`, `interpretability/`, `config/`: **AGPL-3.0** (copyleft)
- `concordia_mini/`: Apache-2.0 (Google DeepMind)
