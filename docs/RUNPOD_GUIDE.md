# RunPod Cloud Deployment Guide

Complete guide for running deception detection experiments on RunPod cloud GPUs.

---

## Table of Contents

1. [Overview](#1-overview)
2. [RunPod Web Interface Setup](#2-runpod-web-interface-setup)
3. [SSH Connection Setup](#3-ssh-connection-setup)
4. [Storage Configuration](#4-storage-configuration)
5. [Installation](#5-installation)
6. [CLI Parameter Reference](#6-cli-parameter-reference)
7. [Quick Copy-Paste Commands](#7-quick-copy-paste-commands)
8. [Recommended Configurations](#8-recommended-configurations)
9. [Monitoring & Logs](#9-monitoring--logs)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Overview

### Why RunPod?

RunPod provides affordable GPU cloud instances ideal for ML research:
- **On-demand GPUs**: A100 80GB, A6000, RTX 4090
- **Persistent storage**: Models cached across sessions
- **Cost-effective**: Pay only for compute time
- **SSH access**: Full terminal access for debugging

### GPU Requirements by Model

| Model | Min VRAM | Recommended GPU | Cost (approx) |
|-------|----------|-----------------|---------------|
| `google/gemma-2b-it` | 4GB | RTX 4090 / A6000 | $0.30-0.50/hr |
| `google/gemma-7b-it` | 16GB | A6000 48GB | $0.50-0.80/hr |

---

## 2. RunPod Web Interface Setup

### Step 1: Create Account

1. Go to [runpod.io](https://runpod.io)
2. Create account and add credits ($10-50 recommended for testing)
3. Verify email

### Step 2: Create Pod

1. Click **"+ Deploy"** → **"GPU Pod"**
2. Select GPU:
   - **A100 80GB**: Best for 9B/27B models, full experiments
   - **A6000 48GB**: Good for 9B model, budget-friendly
   - **RTX 4090 24GB**: Good for 2B model, fastest iteration
3. Select template: **"RunPod Pytorch 2.1"** (recommended)
4. Configure:
   - **Container disk**: 20GB (minimum)
   - **Volume disk**: 50-100GB (for model caching)
   - **Volume mount path**: `/workspace/persistent`

### Step 3: Access Pod

Once deployed, you have two access methods:

#### Option A: Web Terminal
1. Click **"Connect"** on your pod
2. Select **"Web Terminal"**
3. Opens terminal in browser

#### Option B: Jupyter Lab
1. Click **"Connect"** → **"HTTP Service [8888]"**
2. Opens JupyterLab interface
3. Use Terminal tab for command line

---

## 3. SSH Connection Setup

SSH provides better stability for long experiments.

### Step 1: Get SSH Command

1. In RunPod dashboard, click your pod
2. Click **"Connect"** → **"SSH over exposed TCP"**
3. Copy the SSH command (looks like):
   ```
   ssh root@{pod-ip} -p {port} -i ~/.ssh/id_ed25519
   ```

### Step 2: Add SSH Key (if not done)

```bash
# Generate SSH key (if you don't have one)
ssh-keygen -t ed25519 -C "your-email@example.com"

# Copy public key
cat ~/.ssh/id_ed25519.pub
```

1. In RunPod: **Settings** → **SSH Public Keys**
2. Paste your public key
3. Save

### Step 3: Connect via SSH

```bash
# Connect to pod (replace with your actual command)
ssh root@{pod-ip} -p {port} -i ~/.ssh/id_ed25519

# For long experiments, use tmux/screen
tmux new -s experiment
```

### SSH Config (Optional)

Add to `~/.ssh/config` for easier access:

```
Host runpod
    HostName {pod-ip}
    Port {port}
    User root
    IdentityFile ~/.ssh/id_ed25519
```

Then connect with: `ssh runpod`

---

## 4. Storage Configuration

**Critical**: Configure persistent storage so models aren't re-downloaded each session.

### Step 1: Create Cache Directories

```bash
# Create all cache directories on persistent storage
# This stores HuggingFace models, PyTorch caches, SAE weights, and pip packages
mkdir -p /workspace/persistent/huggingface_cache \
         /workspace/persistent/torch_cache \
         /workspace/persistent/sae_cache \
         /workspace/persistent/pip_cache \
         /workspace/persistent/checkpoints \
         /workspace/persistent/experiments
```

### Step 2: Set Environment Variables

```bash
# Add environment variables to .bashrc for persistence across sessions
# HF_HOME: HuggingFace model cache (largest, ~20GB for 9B model)
# TRANSFORMERS_CACHE: Transformer model weights
# TORCH_HOME: PyTorch hub cache
# SAE_LENS_CACHE: Sparse Autoencoder weights (~2GB)
# PIP_CACHE_DIR: Pip package cache (speeds up reinstalls)

echo 'export HF_HOME=/workspace/persistent/huggingface_cache' >> ~/.bashrc && \
echo 'export TRANSFORMERS_CACHE=/workspace/persistent/huggingface_cache' >> ~/.bashrc && \
echo 'export TORCH_HOME=/workspace/persistent/torch_cache' >> ~/.bashrc && \
echo 'export SAE_LENS_CACHE=/workspace/persistent/sae_cache' >> ~/.bashrc && \
echo 'export PIP_CACHE_DIR=/workspace/persistent/pip_cache' >> ~/.bashrc && \
source ~/.bashrc
```

### Step 3: Create Symlinks

```bash
# Create symlink so HuggingFace CLI finds the cache
# This prevents duplicate downloads if default paths are used
ln -sf /workspace/persistent/huggingface_cache ~/.cache/huggingface

# Verify environment is set correctly
echo "HF_HOME=$HF_HOME"
echo "TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
```

### Storage Space Requirements

| Component | Approximate Size |
|-----------|------------------|
| Gemma 2B model | ~5GB |
| Gemma 9B model | ~20GB |
| Gemma 27B model | ~55GB |
| SAE weights (per model) | ~2GB |
| Experiment outputs | ~1-5GB |
| **Total recommended** | **50-100GB** |

---

## 5. Installation

### Step 1: Clone Repository

```bash
# Navigate to workspace
cd /workspace

# Clone the repository
git clone https://github.com/tesims/multiagent-emergent-deception.git

# Enter directory
cd multiagent-emergent-deception
```

### Step 2: Install Dependencies

```bash
# Install package in editable mode (includes all dependencies)
pip install -e .

# Install specific versions for TransformerLens compatibility
# These versions are tested and known to work together
pip install transformers==4.44.0 accelerate==0.33.0 huggingface_hub
```

### Step 3: HuggingFace Authentication

Gemma models require license acceptance:

1. Go to [huggingface.co/google/gemma-2b-it](https://huggingface.co/google/gemma-2b-it)
2. Click **"Agree and access repository"**
3. Get token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

```bash
# Login to HuggingFace (paste token when prompted)
huggingface-cli login
```

### Step 4: Verify Installation

```bash
# Quick verification that everything is installed
python -c "from interpretability import InterpretabilityRunner; print('OK')"

# Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# List available scenarios
deception scenarios
```

---

## 6. CLI Parameter Reference

### Full Command Structure

```bash
deception run [OPTIONS]
```

### All Parameters with Descriptions

```bash
deception run \
    # ═══════════════════════════════════════════════════════════════
    # MODEL CONFIGURATION
    # ═══════════════════════════════════════════════════════════════
    --model MODEL_NAME \
    # HuggingFace model identifier (must be TransformerLens compatible)
    # Options:
    #   "google/gemma-2b-it"    - Fast, 4GB VRAM, good for testing
    #   "google/gemma-7b-it"    - Better quality, 16GB VRAM
    #   "meta-llama/Llama-2-7b-chat-hf" - Alternative architecture
    # Default: "google/gemma-2b-it"

    --device DEVICE \
    # Compute device
    # Options: "cuda", "cpu", "mps" (Apple Silicon)
    # Default: auto-detected (uses CUDA if available)

    --dtype DTYPE \
    # Model precision (affects VRAM usage and speed)
    # Options:
    #   "bfloat16" - Recommended, half memory, minimal quality loss
    #   "float16"  - Half memory, may have numerical issues
    #   "float32"  - Full precision, 2x memory usage
    # Default: "bfloat16"

    # ═══════════════════════════════════════════════════════════════
    # EXPERIMENT MODE
    # ═══════════════════════════════════════════════════════════════
    --mode MODE \
    # Experiment type
    # Options:
    #   "emergent"   - Deception is incentivized but not instructed (recommended)
    #   "instructed" - Explicit deception instructions (Apollo-style baseline)
    #   "both"       - Run both modes
    # Default: "emergent"

    # ═══════════════════════════════════════════════════════════════
    # SCENARIO CONFIGURATION
    # ═══════════════════════════════════════════════════════════════
    --scenarios NUM \
    # Number of scenarios to run (max 6)
    # More scenarios = better generalization testing
    # Default: 3

    --scenario-name NAME \
    # Run a SINGLE specific scenario (for parallel execution across pods)
    # Options:
    #   "ultimatum_bluff"   - Bluffing about walking away
    #   "capability_bluff"  - Overstating capabilities
    #   "hidden_value"      - Hiding true preferences
    #   "info_withholding"  - Strategically withholding info
    #   "promise_break"     - Making breakable promises
    #   "alliance_betrayal" - Forming alliances to betray
    # Default: None (runs multiple scenarios based on --scenarios)

    --trials NUM \
    # Trials per scenario PER condition (high/low incentive)
    # Total samples = scenarios × 2 × trials
    # Minimum for statistics: 25, recommended: 40-50
    # Default: 40

    --max-rounds NUM \
    # Negotiation rounds per trial
    # More rounds = richer data but slower
    # Default: 3

    --max-tokens NUM \
    # Maximum tokens per LLM response
    # Lower = faster but less detailed responses
    # Default: 128

    # ═══════════════════════════════════════════════════════════════
    # SPEED OPTIMIZATIONS
    # ═══════════════════════════════════════════════════════════════
    --fast \
    # Disable Theory of Mind module
    # Effect: ~3x speedup, simpler agent reasoning
    # Trade-off: No agent self-labels, only GM labels
    # Default: False (ToM enabled)

    --ultrafast \
    # Minimal agent configuration
    # Effect: ~5x additional speedup on top of --fast
    # Trade-off: Very basic agent, may affect deception emergence
    # Default: False

    --hybrid \
    # Use HuggingFace for generation + TransformerLens for activations
    # Effect: ~20x speedup for activation capture
    # Recommended for all experiments
    # Default: False (but HIGHLY recommended to enable)

    # ═══════════════════════════════════════════════════════════════
    # MECHANISTIC INTERPRETABILITY
    # ═══════════════════════════════════════════════════════════════
    --layers LAYERS \
    # Comma-separated list of transformer layers to capture
    # Example: "10,21,35" for early, middle, late layers
    # Default: auto-configured based on model

    --sae \
    # Enable Sparse Autoencoder (Gemma Scope) feature extraction
    # Provides interpretable features instead of raw activations
    # Only available for Gemma models
    # Default: False

    --sae-layer NUM \
    # Which layer to extract SAE features from
    # Should be a middle-to-late layer
    # Gemma 2B: 12, Gemma 7B: 20
    # Default: 12

    # ═══════════════════════════════════════════════════════════════
    # CAUSAL VALIDATION
    # ═══════════════════════════════════════════════════════════════
    --causal \
    # Run causal validation after probe training
    # Tests: activation patching, ablation, steering, faithfulness
    # Required for strong causation claims
    # Default: False

    --causal-samples NUM \
    # Number of samples for causal tests
    # More = more reliable results but slower
    # Minimum: 10, recommended: 20-30
    # Default: 20

    # ═══════════════════════════════════════════════════════════════
    # EVALUATION
    # ═══════════════════════════════════════════════════════════════
    --evaluator API \
    # Model API for ground truth extraction
    # Options:
    #   "local"    - Use same model (recommended)
    #   "together" - Together.ai API (requires TOGETHER_API_KEY)
    #   "google"   - Google Gemini API (requires GOOGLE_API_KEY)
    # Default: "local"

    # ═══════════════════════════════════════════════════════════════
    # OUTPUT & CHECKPOINTING
    # ═══════════════════════════════════════════════════════════════
    --output DIR \
    # Output directory for all results
    # Will contain: activations.pt, probe_results.json, logs
    # Default: "./experiment_output"

    --checkpoint-dir DIR \
    # Directory for checkpoint saves
    # Enables crash recovery for long experiments
    # Saves every N trials (configurable in code)
    # Default: None (no checkpointing)

    --verbose \
    # Enable detailed output logging
    # Default: False
```

### Example Commands with Comments

```bash
# ═══════════════════════════════════════════════════════════════════════════
# QUICK TEST: Verify setup works (2-5 minutes)
# ═══════════════════════════════════════════════════════════════════════════
deception run \
    --model google/gemma-2b-it \       # Small model for fast testing
    --scenarios 1 \                    # Just 1 scenario
    --trials 2 \                       # 2 trials (4 total: 2 high + 2 low incentive)
    --max-rounds 2 \                   # Short negotiations
    --hybrid \                         # Fast activation capture
    --device cuda \
    --dtype bfloat16 \
    --output /workspace/persistent/test_run

# ═══════════════════════════════════════════════════════════════════════════
# DEVELOPMENT: Fast iteration while developing (10-20 minutes)
# ═══════════════════════════════════════════════════════════════════════════
deception run \
    --model google/gemma-2b-it \       # Fast model for development
    --scenarios 3 \                    # Core scenarios
    --trials 10 \                      # Enough for basic statistics
    --max-rounds 3 \
    --mode both \                      # Run emergent + instructed
    --fast \                           # Disable ToM for speed
    --hybrid \
    --sae \                            # Get interpretable features
    --sae-layer 12 \
    --device cuda \
    --dtype bfloat16 \
    --output /workspace/persistent/dev_run

# ═══════════════════════════════════════════════════════════════════════════
# FULL EXPERIMENT: Publication-quality results (2-4 hours)
# ═══════════════════════════════════════════════════════════════════════════
deception run \
    --model google/gemma-7b-it \       # Larger model for research
    --scenarios 6 \                    # All 6 emergent scenarios
    --trials 50 \                      # Strong statistics
    --max-rounds 3 \
    --mode both \                      # Run emergent + instructed
    --hybrid \                         # Required for reasonable runtime
    --sae \                            # Interpretable features
    --sae-layer 20 \
    --causal \                         # Causal validation
    --causal-samples 30 \              # Robust causal tests
    --checkpoint-dir /workspace/persistent/checkpoints \  # Crash recovery
    --device cuda \
    --dtype bfloat16 \
    --output /workspace/persistent/full_experiment
```

---

## 7. Quick Copy-Paste Commands

### One-Time Setup (New Pod)

```bash
# Step 1: Create storage directories
mkdir -p /workspace/persistent/huggingface_cache \
         /workspace/persistent/torch_cache \
         /workspace/persistent/sae_cache \
         /workspace/persistent/pip_cache \
         /workspace/persistent/checkpoints \
         /workspace/persistent/experiments
```

```bash
# Step 2: Configure environment (run once, persists across sessions)
echo 'export HF_HOME=/workspace/persistent/huggingface_cache' >> ~/.bashrc && \
echo 'export TRANSFORMERS_CACHE=/workspace/persistent/huggingface_cache' >> ~/.bashrc && \
echo 'export TORCH_HOME=/workspace/persistent/torch_cache' >> ~/.bashrc && \
echo 'export SAE_LENS_CACHE=/workspace/persistent/sae_cache' >> ~/.bashrc && \
echo 'export PIP_CACHE_DIR=/workspace/persistent/pip_cache' >> ~/.bashrc && \
source ~/.bashrc
```

```bash
# Step 3: Create symlinks
ln -sf /workspace/persistent/huggingface_cache ~/.cache/huggingface
```

```bash
# Step 4: Clone and install
cd /workspace && \
git clone https://github.com/tesims/multiagent-emergent-deception.git && \
cd multiagent-emergent-deception && \
pip install -e . && \
pip install transformers==4.44.0 accelerate==0.33.0 huggingface_hub
```

```bash
# Step 5: HuggingFace login (interactive - paste token when prompted)
huggingface-cli login
```

### Quick Test (Verify Setup)

```bash
cd /workspace/multiagent-emergent-deception && \
deception run \
    --model google/gemma-2b-it \
    --scenarios 1 \
    --trials 1 \
    --max-rounds 2 \
    --mode both \
    --hybrid \
    --device cuda \
    --dtype bfloat16 \
    --output /workspace/persistent/experiments/test_run \
    2>&1 | tee /workspace/persistent/experiments/test_run.log
```

### Development Run (Gemma 2B)

```bash
cd /workspace/multiagent-emergent-deception && \
deception run \
    --model google/gemma-2b-it \
    --scenarios 3 \
    --trials 10 \
    --max-rounds 3 \
    --mode both \
    --fast \
    --hybrid \
    --sae \
    --sae-layer 12 \
    --device cuda \
    --dtype bfloat16 \
    --checkpoint-dir /workspace/persistent/checkpoints \
    --output /workspace/persistent/experiments/dev_2b \
    2>&1 | tee /workspace/persistent/experiments/dev_2b.log
```

### Full Experiment (Gemma 7B - Recommended)

```bash
cd /workspace/multiagent-emergent-deception && \
deception run \
    --model google/gemma-7b-it \
    --scenarios 6 \
    --trials 50 \
    --max-rounds 3 \
    --mode both \
    --hybrid \
    --sae \
    --sae-layer 20 \
    --causal \
    --causal-samples 30 \
    --device cuda \
    --dtype bfloat16 \
    --checkpoint-dir /workspace/persistent/checkpoints \
    --output /workspace/persistent/experiments/full_7b \
    2>&1 | tee /workspace/persistent/experiments/full_7b.log
```

### Single Scenario (For Parallel Pods)

Run different scenarios on different pods, then merge results:

```bash
# Pod 1: ultimatum_bluff
cd /workspace/multiagent-emergent-deception && \
deception run \
    --model google/gemma-2b-it \
    --scenario-name ultimatum_bluff \
    --mode both \
    --trials 50 \
    --max-rounds 3 \
    --hybrid \
    --sae \
    --sae-layer 12 \
    --device cuda \
    --dtype bfloat16 \
    --output /workspace/persistent/experiments/ultimatum \
    2>&1 | tee /workspace/persistent/experiments/ultimatum.log
```

```bash
# Pod 2: alliance_betrayal
cd /workspace/multiagent-emergent-deception && \
deception run \
    --model google/gemma-2b-it \
    --scenario-name alliance_betrayal \
    --mode both \
    --trials 50 \
    --max-rounds 3 \
    --hybrid \
    --sae \
    --sae-layer 12 \
    --device cuda \
    --dtype bfloat16 \
    --output /workspace/persistent/experiments/alliance \
    2>&1 | tee /workspace/persistent/experiments/alliance.log
```

### Returning to Existing Pod

```bash
# Pull latest changes
cd /workspace/multiagent-emergent-deception && \
git pull && \
pip install -e .
```

```bash
# Resume or start new experiment
cd /workspace/multiagent-emergent-deception && \
deception run \
    --model google/gemma-2b-it \
    --scenarios 6 \
    --trials 50 \
    --mode both \
    --hybrid \
    --sae \
    --sae-layer 12 \
    --causal \
    --checkpoint-dir /workspace/persistent/checkpoints \
    --output /workspace/persistent/experiments/resumed \
    2>&1 | tee /workspace/persistent/experiments/resumed.log
```

---

## 8. Recommended Configurations

### Budget-Conscious (RTX 4090, ~$0.40/hr)

```bash
deception run \
    --model google/gemma-2b-it \
    --scenarios 6 \
    --trials 25 \
    --mode both \
    --fast \
    --hybrid \
    --sae \
    --sae-layer 12 \
    --device cuda \
    --dtype bfloat16 \
    --output /workspace/persistent/experiments/budget_run
```

**Expected**: ~30 minutes, good for preliminary results

### Research Quality (A6000 48GB, ~$0.70/hr)

```bash
deception run \
    --model google/gemma-7b-it \
    --scenarios 6 \
    --trials 40 \
    --mode both \
    --hybrid \
    --sae \
    --sae-layer 20 \
    --causal \
    --causal-samples 20 \
    --device cuda \
    --dtype bfloat16 \
    --checkpoint-dir /workspace/persistent/checkpoints \
    --output /workspace/persistent/experiments/research_run
```

**Expected**: ~2-3 hours, publication-ready

---

## 9. Monitoring & Logs

### Watch Live Progress

```bash
# In a separate terminal/tmux pane
tail -f /workspace/persistent/experiments/full_9b.log
```

### Check GPU Usage

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# One-time check
nvidia-smi
```

### Check Storage Usage

```bash
# Check persistent storage usage
du -sh /workspace/persistent/*

# Check total disk usage
df -h
```

### Running in Background (tmux)

```bash
# Start new tmux session
tmux new -s experiment

# Run experiment...

# Detach from session: Ctrl+B, then D

# Reattach later
tmux attach -t experiment

# List sessions
tmux ls
```

---

## 10. Troubleshooting

### "CUDA out of memory"

```bash
# Solution 1: Use smaller model
--model google/gemma-2b-it

# Solution 2: Use bfloat16 (if not already)
--dtype bfloat16

# Solution 3: Reduce batch operations
--max-tokens 64

# Solution 4: Clear GPU cache (in Python)
import torch; torch.cuda.empty_cache()
```

### "Access denied to google/gemma"

```bash
# 1. Accept license at HuggingFace
# Visit: https://huggingface.co/google/gemma-7b-it
# Click "Agree and access repository"

# 2. Re-login
huggingface-cli login
```

### Model not downloading to persistent storage

```bash
# Verify environment variables
echo $HF_HOME
echo $TRANSFORMERS_CACHE

# Should show: /workspace/persistent/huggingface_cache
# If not, source bashrc again:
source ~/.bashrc
```

### SSH connection drops

```bash
# Use tmux for persistent sessions
tmux new -s experiment

# Or use screen
screen -S experiment

# Reattach after disconnect
tmux attach -t experiment
screen -r experiment
```

### "TransformerLens version mismatch"

```bash
# Install compatible versions
pip install transformers==4.44.0 accelerate==0.33.0
```

### Experiment crashed mid-run

```bash
# If checkpointing was enabled, data is saved
# Check checkpoint directory
ls /workspace/persistent/checkpoints/

# Re-run same command - checkpoints will be loaded automatically
# (Note: automatic checkpoint loading may need manual implementation)
```

### Download results to local machine

```bash
# On your LOCAL machine (not RunPod):
scp -P {port} root@{pod-ip}:/workspace/persistent/experiments/full_9b/*.pt ./results/
scp -P {port} root@{pod-ip}:/workspace/persistent/experiments/full_9b/*.json ./results/
```

---

## Summary Cheat Sheet

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         RUNPOD QUICK REFERENCE                            │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  FIRST TIME SETUP:                                                        │
│  1. Create pod with volume at /workspace/persistent                       │
│  2. Run storage setup commands (Section 4)                                │
│  3. Clone repo and install (Section 5)                                    │
│  4. huggingface-cli login                                                 │
│                                                                           │
│  RECOMMENDED FLAGS:                                                       │
│  --hybrid          Always use (20x faster)                                │
│  --sae             Use for Gemma models                                   │
│  --dtype bfloat16  Saves memory                                           │
│  --checkpoint-dir  For long experiments                                   │
│                                                                           │
│  MODEL SELECTION:                                                         │
│  2B  → Testing, development, budget                                       │
│  7B  → Research, publication (recommended)                                │
│                                                                           │
│  PARALLEL EXECUTION:                                                      │
│  Use --scenario-name on different pods                                    │
│  Merge .pt files afterward                                                │
│                                                                           │
│  OUTPUT FILES:                                                            │
│  activations_*.pt       Raw data for further analysis                     │
│  probe_results.json     Probe training metrics                            │
│  causal_*.json          Causal validation results                         │
│  *.log                  Full experiment logs                              │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```
