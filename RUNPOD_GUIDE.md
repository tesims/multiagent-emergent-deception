# RunPod Experiment Guide

## Quick Setup (A100 80GB)

### Step 1: Storage Setup
```bash
mkdir -p /workspace/persistent/huggingface_cache /workspace/persistent/torch_cache /workspace/persistent/sae_cache
```

```bash
export HF_HOME=/workspace/persistent/huggingface_cache
export TORCH_HOME=/workspace/persistent/torch_cache
export SAE_LENS_CACHE=/workspace/persistent/sae_cache
```

### Step 2: Install
```bash
cd /workspace
git clone https://github.com/tesims/multiagent-emergent-deception.git
cd multiagent-emergent-deception
pip install -e .
```

### Step 3: HuggingFace Login
```bash
huggingface-cli login
```

### Step 4: Run Experiment

**Quick Test (1 scenario, 1 trial):**
```bash
python -m interpretability.run_deception_experiment \
    --mode emergent \
    --scenario-name alliance_betrayal \
    --trials 1 \
    --device cuda
```

**Full Experiment (6 scenarios, 50 trials):**
```bash
python -m interpretability.run_deception_experiment \
    --mode emergent \
    --scenarios 6 \
    --trials 50 \
    --hybrid --sae --causal \
    --device cuda \
    --output /workspace/persistent/results
```

**With Gemma 2B (faster, less VRAM):**
```bash
python -m interpretability.run_deception_experiment \
    --model google/gemma-2-2b-it \
    --mode emergent \
    --scenarios 6 \
    --trials 10 \
    --hybrid --sae --causal \
    --device cuda
```

## Key Differences from Old Repo

| Old (gsoc-concordia) | New (standalone) |
|---------------------|------------------|
| `cd concordia/prefabs/entity/negotiation/evaluation` | Just `cd multiagent-emergent-deception` |
| `python run_deception_experiment.py` | `python -m interpretability.run_deception_experiment` |
| Complex nested imports | Clean top-level packages |

## Available Options

```
--model         Model name (default: google/gemma-2-9b-it)
--mode          emergent, instructed, or contest
--scenarios     Number of scenarios (1-6)
--scenario-name Specific scenario name
--trials        Number of trials per scenario
--hybrid        Enable activation capture
--sae           Enable SAE analysis
--causal        Enable causal validation
--device        cuda or cpu
--output        Output directory
```
