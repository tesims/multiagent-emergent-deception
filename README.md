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

```bash
# Install
git clone https://github.com/tesims/multiagent-emergent-deception.git
cd multiagent-emergent-deception
pip install -e .

# Run experiment
python -m interpretability.run_deception_experiment --mode emergent --trials 10
```

## Configuration-Driven Design

All experiments are configured through `config/`:

```python
from config import ExperimentConfig, ModelConfig, ScenarioConfig

# Customize your experiment
experiment = ExperimentConfig(
    model=ModelConfig(
        name="google/gemma-2-2b-it",
        device="cuda",
        use_sae=True,
    ),
    scenarios=ScenarioConfig(
        mode="emergent",
        scenarios=["ultimatum_bluff", "alliance_betrayal"],
        num_trials=50,
    ),
)
```

### Available Configs

| Config | Purpose |
|--------|---------|
| `ExperimentConfig` | Main experiment settings |
| `ModelConfig` | LLM, TransformerLens, SAE settings |
| `ProbeConfig` | Linear probe training |
| `CausalConfig` | Activation patching, ablation, steering |
| `ScenarioConfig` | Deception scenarios and trials |
| `StrategyConfig` | Agent behavior (negotiation) |

### Preset Configurations

```python
from config.experiment import QUICK_TEST, FULL_EXPERIMENT, FAST_ITERATION

# Quick validation (1 scenario, 1 trial)
experiment = QUICK_TEST

# Full research run (6 scenarios, 50 trials)
experiment = FULL_EXPERIMENT

# Fast iteration (no SAE, minimal causal)
experiment = FAST_ITERATION
```

## Project Structure

```
multiagent-emergent-deception/
├── config/                    # CENTRAL CONFIGURATION
│   ├── experiment.py          # Core experiment configs
│   └── agents/
│       └── negotiation.py     # Negotiation agent configs
│
├── interpretability/          # DECEPTION DETECTION PIPELINE
│   ├── run_deception_experiment.py  # Main entry point
│   ├── evaluation.py          # HybridLanguageModel, runner
│   ├── scenarios/             # Deception scenarios
│   ├── probes/                # Probe training & SAE tools
│   └── causal/                # Causal validation
│
├── negotiation/               # AGENT IMPLEMENTATION (extensible)
│   ├── components/            # 6 cognitive modules
│   └── game_master/           # GM components
│
└── concordia_mini/            # Framework dependency
```

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

### Contest (Game-Theoretic)
Complex multi-agent scenarios: Fishery, Treaty, Gameshow.

## Extending for New Agent Types

The framework supports agents beyond negotiation:

```python
# 1. Create config/agents/your_agent.py
@dataclass
class YourAgentConfig:
    behavior_param: float = 0.5
    ...

# 2. Create agents/your_agent/
#    with components matching your research needs

# 3. Add scenarios in interpretability/scenarios/
```

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
