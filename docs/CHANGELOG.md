# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-10

Initial public release of the Multi-Agent Emergent Deception framework.

### Core Features

- **Emergent Deception Scenarios**: 6 negotiation scenarios where deception is incentivized but never instructed
  - `ultimatum_bluff` - Bluffing about walking away from negotiation
  - `capability_bluff` - Overstating capabilities or resources
  - `hidden_value` - Hiding true preferences to gain advantage
  - `info_withholding` - Strategically withholding information
  - `promise_break` - Making promises with intent to break
  - `alliance_betrayal` - Forming alliances only to betray

- **Mechanistic Interpretability Pipeline**
  - TransformerLens integration for activation capture
  - SAE (Sparse Autoencoder) feature extraction via Gemma Scope
  - Linear probe training (Ridge, mass-mean)
  - Cross-scenario generalization testing

- **Causal Validation Suite**
  - Activation patching
  - Ablation testing
  - Steering vector analysis
  - Probe faithfulness tests
  - Selectivity tests

- **Multi-Agent Framework**
  - Concordia-based agent implementation
  - Theory of Mind module with emotional state detection
  - Game Master for objective ground truth labeling
  - Configurable agent modules

### Configuration System

- `ExperimentConfig` - Main experiment configuration
- `ModelConfig` - LLM and TransformerLens settings
- `ProbeConfig` - Linear probe training parameters
- `CausalConfig` - Causal validation settings
- `ScenarioConfig` - Scenario and trial configuration
- `MODEL_PRESETS` - Auto-configuration for supported models

### Supported Models

| Model | VRAM | SAE Support |
|-------|------|-------------|
| `google/gemma-2b-it` | ~4GB | Yes |
| `google/gemma-7b-it` | ~20GB | Yes |
| `google/gemma-7b-it` | ~54GB | Yes |
| `meta-llama/Llama-3.1-8B-Instruct` | ~16GB | No |

### CLI Interface

- `deception run` - Run complete experiment pipeline
- `deception train` - Train probes on existing data
- `deception scenarios` - List available scenarios

### Documentation

- `README.md` - Quick start and overview
- `docs/SETUP.md` - Detailed installation and configuration
- `docs/METHODOLOGY.md` - Technical methodology and research background
- `docs/OUTPUT_GUIDE.md` - How to interpret experiment outputs
- `docs/ARCHITECTURE.md` - System architecture with diagrams
- `docs/RUNPOD_GUIDE.md` - Cloud GPU deployment guide
- `docs/CONTRIBUTING.md` - Contribution guidelines

### Testing

- Unit tests for configuration validation
- Unit tests for probe training
- Graceful handling of missing PyTorch

### License

- `negotiation/`, `interpretability/`, `config/`: AGPL-3.0
- `concordia_mini/`: Apache-2.0 (Google DeepMind)
