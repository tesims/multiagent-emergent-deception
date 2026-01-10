# Multi-Agent Emergent Deception

Mechanistic interpretability for detecting emergent deception in LLM negotiation agents.

## Overview

This project provides:
1. **Multi-Agent Negotiation Framework** - Cognitive agent modules for multi-agent negotiation simulations
2. **Interpretability Pipeline** - Tools for detecting deception via activation analysis and linear probes
3. **Emergent Deception Scenarios** - 6 incentive-based scenarios that elicit deception without explicit instructions

## Key Features

- **HybridLanguageModel**: Combines HuggingFace generation with TransformerLens activation capture
- **6 Deception Scenarios**: Ultimatum bluff, capability bluff, hidden value, info withholding, promise break, alliance betrayal
- **Linear Probes**: Train probes to detect deception from model activations
- **Causal Validation**: Activation patching, ablation, and steering vector tests
- **Sparse Autoencoder Analysis**: Integration with Gemma Scope SAEs

## Installation

```bash
git clone https://github.com/tesims/multiagent-emergent-deception.git
cd multiagent-emergent-deception
pip install -e .
```

## Quick Start

```python
# Run deception detection experiment
python -m interpretability.run_deception_experiment \
    --mode emergent \
    --scenarios 6 \
    --trials 10 \
    --model google/gemma-2-2b-it \
    --device cuda
```

## Project Structure

```
multiagent-emergent-deception/
├── concordia_mini/     # Minimal Concordia framework (Google DeepMind)
├── negotiation/        # Cognitive negotiation agent modules
│   ├── components/     # 6 cognitive modules (ToM, cultural, temporal, etc.)
│   └── game_master/    # Game master components
├── interpretability/   # Deception detection pipeline
│   ├── run_deception_experiment.py
│   ├── interpretability_evaluation.py
│   ├── train_probes.py
│   └── causal_validation.py
└── docs/               # Documentation
```

## Cognitive Modules

| Module | Description |
|--------|-------------|
| Theory of Mind | Recursive belief modeling, emotion tracking |
| Cultural Adaptation | Hofstede cultural dimensions |
| Temporal Strategy | Multi-horizon planning |
| Swarm Intelligence | Multi-agent voting |
| Uncertainty Aware | Bayesian belief updates |
| Strategy Evolution | Genetic algorithms for strategy optimization |

## Deception Scenarios

1. **Ultimatum Bluff** - False claims about minimum acceptable offers
2. **Capability Bluff** - Overstating capabilities
3. **Hidden Value** - Misrepresenting true valuations
4. **Info Withholding** - Hiding known defects
5. **Promise Break** - Signaling defection while promising cooperation
6. **Alliance Betrayal** - Reassuring allies while considering betrayal

## Citation

```bibtex
@software{multiagent_emergent_deception,
  title = {Multi-Agent Emergent Deception: Mechanistic Interpretability for LLM Negotiation},
  author = {Sims, Teanna},
  year = {2025},
  url = {https://github.com/tesims/multiagent-emergent-deception}
}
```

## License

Apache 2.0 - See LICENSE file.

## Acknowledgments

This project builds upon [Concordia](https://github.com/google-deepmind/concordia)
by Google DeepMind, included under Apache 2.0 license.
