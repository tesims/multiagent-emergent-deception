"""Basic example of running a deception detection experiment."""

from interpretability.emergent_prompts import EMERGENT_SCENARIOS
from interpretability.interpretability_evaluation import EvaluationConfig

# Show available scenarios
print("Available deception scenarios:")
for name, scenario in EMERGENT_SCENARIOS.items():
    print(f"  - {name}: {scenario['description'][:60]}...")

# Example configuration
config = EvaluationConfig(
    model_name="google/gemma-2-2b-it",
    num_trials=5,
    device="cuda",  # or "cpu"
)

print(f"\nExample config: {config}")

# To run full experiment:
# python -m interpretability.run_deception_experiment --mode emergent --trials 5
