# Code Review Checklist

Prioritized by impact level (high → low) to avoid backtracking.

---

## TIER 1: CRITICAL (Must fix - blocks functionality)

### 1.1 [ ] Fix broken example - basic_experiment.py
**File:** `examples/basic_experiment.py`
**Issue:**
- `scenario['description']` doesn't exist (scenarios have nested IncentiveCondition keys)
- `EvaluationConfig` imported from wrong location
**Impact:** Bad first impression, example crashes
**Fix:** Rewrite example to use actual scenario structure

### 1.2 [ ] Add config* to pyproject.toml
**File:** `pyproject.toml:34`
**Issue:** `include = ["concordia_mini*", "negotiation*", "interpretability*"]` missing `config*`
**Impact:** `pip install -e .` won't include config package
**Fix:** Add `"config*"` to include list

### 1.3 [ ] Fix mutable class attributes in StrategyConfig
**File:** `config/agents/negotiation.py:30-90`
**Issue:** `StrategyConfig` uses mutable class attributes that can be modified globally
**Impact:** Cross-test contamination, unpredictable behavior
**Example bug:**
```python
# File A modifies global state
StrategyConfig.COOPERATIVE_ACCEPTANCE_THRESHOLD = 0.90
# File B expects default - BROKEN
```
**Fix:** Convert to frozen dataclass or use instance attributes

---

## TIER 2: MAJOR (Should fix - causes bugs/confusion)

### 2.1 [ ] Fix PCA data leakage in train_probes.py
**File:** `interpretability/probes/train_probes.py:78`
**Issue:** `X_pca = pca.fit_transform(X)` fits PCA on ALL data including test set
**Impact:** Inflated probe performance metrics (data leakage)
**Fix:** Use `pca.transform(X)` after fitting only on X_train

### 2.2 [ ] Remove debug print statements
**File:** `interpretability/evaluation.py:946` and others
**Issue:** `print(f"  [DEBUG]...")` throughout production code
**Impact:** Noisy output, unprofessional
**Fix:** Replace with proper logging module or remove

### 2.3 [ ] Fix bare exception handling
**File:** `interpretability/evaluation.py:535-538`
**Issue:** `except Exception as e: print(...)` swallows errors
**Impact:** Hidden failures, hard to debug
**Fix:** Re-raise or log properly with traceback

### 2.4 [ ] Fix silent parse failures in theory_of_mind.py
**File:** `negotiation/components/theory_of_mind.py:144-147`
**Issue:** `except (ValueError, IndexError): pass` silently ignores errors
**Impact:** Lost data, no indication of parse failures
**Fix:** Log warnings or track failure counts

### 2.5 [ ] Add upper version bounds to dependencies
**File:** `pyproject.toml:16-24`
**Issue:** `torch>=2.0` allows any future version
**Impact:** Future breaking changes
**Fix:** Add upper bounds: `torch>=2.0.0,<3.0.0`

### 2.6 [ ] Remove redundant setup.py
**File:** `setup.py`
**Issue:** Duplicates pyproject.toml, no metadata
**Impact:** Confusion about which to use
**Fix:** Delete setup.py, use pyproject.toml only

---

## TIER 3: IMPORTANT (Research methodology)

### 3.1 [ ] Add multiple random seeds
**Files:** Throughout codebase
**Issue:** `random_state=42` used everywhere
**Impact:** Results may not be reproducible/generalizable
**Fix:** Parameterize seed, run experiments with seeds [42, 123, 456, 789, 1337]

### 3.2 [ ] Document token position for probing
**File:** `interpretability/evaluation.py`
**Issue:** Unclear whether last token, all tokens, or specific positions
**Impact:** Reproducibility, comparison to other work
**Fix:** Add explicit documentation and config option

### 3.3 [ ] Add cross-scenario generalization test
**File:** `interpretability/probes/train_probes.py`
**Issue:** Train on scenarios A-E, test on F not clearly implemented
**Impact:** May be detecting scenario-specific artifacts
**Fix:** Add leave-one-scenario-out evaluation with clear reporting

### 3.4 [ ] Fix binary vs continuous label handling
**File:** `interpretability/probes/train_probes.py:97-99`
**Issue:** Uses 0.5 threshold without justification
**Impact:** Loss of information, arbitrary cutoff
**Fix:** Add threshold sensitivity analysis or use continuous regression

---

## TIER 4: MODERATE (Code quality)

### 4.1 [ ] Add unit tests for core functions
**File:** `tests/`
**Issue:** Only import tests exist
**Missing tests for:**
- Ground truth detection logic
- Probe training functions
- Causal validation
- Scenario parsing
**Fix:** Add pytest tests for each module

### 4.2 [ ] Add __version__ to config package
**File:** `config/__init__.py`
**Issue:** No version exported
**Fix:** Add `__version__ = "1.0.0"`

### 4.3 [ ] Fix type hints - reduce Any usage
**Files:** `baseline_agents.py:173`, `theory_of_mind.py`
**Issue:** `model: Any` should be `model: Optional[LanguageModel]`
**Fix:** Add proper type hints

### 4.4 [ ] Replace Dict[str, Any] returns with dataclasses
**Files:** `evaluation.py`, `train_probes.py`
**Issue:** No type safety on return values
**Fix:** Create typed result dataclasses

---

## TIER 5: MINOR (Nice to have)

### 5.1 [ ] Add validation to config values
**File:** `config/experiment.py:74`
**Issue:** `train_ratio` should validate 0 < x < 1
**Fix:** Add Pydantic or manual validation

### 5.2 [ ] Pre-compile regex patterns
**File:** `interpretability/scenarios/emergent_prompts.py:443,464`
**Issue:** `re.findall()` called without pre-compiled patterns
**Fix:** Compile patterns at module level

### 5.3 [ ] Add CHANGELOG.md
**Issue:** No version history
**Fix:** Create CHANGELOG.md documenting changes

### 5.4 [ ] Add CI pipeline (GitHub Actions)
**Issue:** No automated testing
**Fix:** Add `.github/workflows/test.yml`

---

## TIER 6: ARCHITECTURAL (Future refactoring)

### 6.1 [ ] Refactor InterpretabilityRunner (2200 lines)
**File:** `interpretability/evaluation.py`
**Issue:** God class with ~30 methods
**Proposed split:**
- `ModelManager` (~300 lines) - Load/manage models
- `TrialRunner` (~400 lines) - Execute individual trials
- `GroundTruthDetector` (~300 lines) - Deception detection logic
- `SampleCollector` (~200 lines) - Activation collection
- `DatasetBuilder` (~200 lines) - Save/load datasets
**Note:** Do this AFTER all other fixes to avoid conflicts

### 6.2 [ ] Consider Pydantic for config validation
**Issue:** Dataclasses lack built-in validation
**Fix:** Migrate to Pydantic BaseModel with validators

### 6.3 [ ] Consider Click for CLI
**File:** `interpretability/run_deception_experiment.py`
**Issue:** 100+ line argparse setup
**Fix:** Migrate to Click with subcommands

---

## VALIDATION CHECKLIST (Run after fixes)

```bash
# 1. Install works
pip install -e .

# 2. Config imports
python -c "from config import ExperimentConfig; print('OK')"

# 3. Example runs
python examples/basic_experiment.py

# 4. Tests pass
pytest tests/ -v

# 5. No import errors
python -c "from interpretability import InterpretabilityRunner; print('OK')"
```

---

## SUMMARY BY TIER

| Tier | Count | Priority |
|------|-------|----------|
| 1. Critical | 3 | MUST fix before release |
| 2. Major | 6 | Should fix before release |
| 3. Important | 4 | Fix for research validity |
| 4. Moderate | 4 | Fix for code quality |
| 5. Minor | 4 | Nice to have |
| 6. Architectural | 3 | Future work |
| **TOTAL** | **24** | |
