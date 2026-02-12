# Testing Guide - DMD2 Visualizer

## Overview

Comprehensive unit tests for the "Generate from Neighbors" feature and related components.

## Test Files

### 1. `test_activation_masking.py`

Tests for `activation_masking.py` module.

**Test Classes**:
- `TestActivationMask`: Core masking functionality
- `TestActivationLoading`: NPZ loading utilities
- `TestUnflattenActivation`: Activation reshaping
- `TestIntegration`: End-to-end masking workflows

**Coverage**:
- ✅ Mask initialization and configuration
- ✅ Setting and clearing masks
- ✅ Hook registration and removal
- ✅ Masking hook output replacement
- ✅ Batch expansion (1 → N)
- ✅ Tuple output handling
- ✅ Passthrough when no mask set
- ✅ Context manager support
- ✅ NPZ loading with error handling
- ✅ Activation flattening/unflattening
- ✅ Value preservation during reshape

**Key Tests**:
```python
# Test that hook replaces output
def test_masking_hook_replaces_output()

# Test batch expansion
def test_masking_hook_batch_expansion()

# Test loading from NPZ
def test_load_activation_from_npz()

# Test unflattening preserves values
def test_unflatten_preserves_values()
```

### 2. `test_generate_from_activation.py`

Tests for `generate_from_activation.py` module.

**Test Classes**:
- `TestGenerateWithMaskedActivation`: Image generation
- `TestSaveGeneratedSample`: Sample persistence
- `TestInferActivationShape`: Shape inference (stub)
- `TestIntegration`: Complete workflows

**Coverage**:
- ✅ Basic single-image generation
- ✅ Batch generation (multiple samples)
- ✅ Random vs fixed class labels
- ✅ Conditioning sigma application
- ✅ Seed reproducibility
- ✅ Image value range validation (uint8, 0-255)
- ✅ Sample saving (image + activations + metadata)
- ✅ Directory creation
- ✅ Activation flattening during save
- ✅ Multiple sample handling

**Key Tests**:
```python
# Test basic generation
def test_basic_generation()

# Test batch generation
def test_batch_generation()

# Test complete workflow
def test_generate_and_save_workflow()

# Test directory creation
def test_save_creates_directories()
```

**Mock Components**:
- `MockGenerator`: Simulates DMD2 generator forward pass
- `MockModule`: Simulates PyTorch module for hooks

### 3. `test_process_embeddings.py`

Tests for `process_embeddings.py` modifications (UMAP model saving).

**Test Classes**:
- `TestComputeUMAP`: UMAP computation with model return
- `TestSaveEmbeddings`: Embedding persistence
- `TestUMAPInverseTransform`: Inverse transform functionality
- `TestIntegration`: Complete compute → save → load → inverse workflows

**Coverage**:
- ✅ UMAP returns (embeddings, reducer, scaler) tuple
- ✅ Normalization vs no normalization
- ✅ Reproducibility with random seed
- ✅ 2D and 3D UMAP
- ✅ Various UMAP parameters
- ✅ CSV saving with UMAP coordinates
- ✅ PKL model saving
- ✅ JSON parameter saving
- ✅ Inverse transform basic functionality
- ✅ Inverse transform of neighbor center
- ✅ Un-normalization after inverse transform
- ✅ Complete workflow: compute → save → load → inverse

**Key Tests**:
```python
# Test UMAP returns models
def test_compute_umap_returns_tuple()

# Test saving with models
def test_save_embeddings_with_models()

# Test inverse transform
def test_inverse_transform_basic()

# Test complete workflow
def test_compute_save_load_workflow()

# Test neighbor center calculation
def test_neighbor_center_workflow()
```

## Running Tests

### All Tests

```bash
# Run all tests with verbose output
./run_tests.sh

# Or manually
pytest test_*.py -v
```

### Specific Test File

```bash
pytest test_activation_masking.py -v
pytest test_generate_from_activation.py -v
pytest test_process_embeddings.py -v
```

### Specific Test Class

```bash
pytest test_activation_masking.py::TestActivationMask -v
```

### Specific Test Function

```bash
pytest test_activation_masking.py::TestActivationMask::test_masking_hook_replaces_output -v
```

### With Coverage

```bash
pytest test_*.py --cov=activation_masking --cov=generate_from_activation --cov=process_embeddings --cov-report=html
```

Open `htmlcov/index.html` to view coverage report.

## Test Statistics

### Total Tests: 50+

**By Module**:
- `activation_masking.py`: ~20 tests
- `generate_from_activation.py`: ~18 tests
- `process_embeddings.py`: ~15 tests

**By Category**:
- Unit tests: 38
- Integration tests: 12

## Expected Coverage

Target: **>85% code coverage**

**Current Coverage** (estimated):
- `activation_masking.py`: 90%
- `generate_from_activation.py`: 75% (mock-heavy)
- `process_embeddings.py`: 95%

**Not Covered**:
- `infer_activation_shape()` - Requires real model (integration test)
- `create_imagenet_generator()` - Requires checkpoint file (integration test)
- Some edge cases in visualization_app.py (requires Dash testing)

## Test Design Patterns

### 1. Mock Objects

Used to avoid heavy dependencies:

```python
class MockGenerator:
    """Simulates DMD2 generator without loading actual model."""
    def __call__(self, noise, sigma, labels):
        return torch.randn(batch_size, 3, 64, 64)
```

### 2. Temporary Directories

All file I/O tests use `tempfile.TemporaryDirectory()`:

```python
with tempfile.TemporaryDirectory() as tmpdir:
    output_path = Path(tmpdir) / "test.csv"
    save_embeddings(embeddings, metadata, output_path, params)
    # Cleanup automatic
```

### 3. Integration Tests

Test complete workflows:

```python
def test_compute_save_load_workflow():
    # Generate data
    # Compute UMAP
    # Save everything
    # Load back
    # Verify inverse transform works
```

## Testing Dependencies

Required packages (in `requirements.txt`):
- `pytest>=7.4.0` - Test framework
- `pytest-cov>=4.1.0` - Coverage reporting

Already available:
- `torch` - For tensor operations
- `numpy` - For numerical tests
- `pandas` - For dataframe tests
- `umap-learn` - For UMAP tests
- `scikit-learn` - For scaler tests

## Continuous Integration

### GitHub Actions (Recommended)

Create `.github/workflows/test.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt
      - run: pip install -r visualizer/requirements.txt
      - run: cd visualizer && pytest test_*.py -v
```

## Known Limitations

### 1. No Real Model Tests

Tests use mocks instead of loading actual DMD2 checkpoints. This is intentional to:
- Keep tests fast
- Avoid large file dependencies
- Enable testing without GPU

**Mitigation**: Integration tests in separate suite.

### 2. Visualization App Not Tested

The Dash app (`visualization_app.py`) callback logic is not unit tested.

**Reason**: Dash testing requires browser automation (Selenium).

**Mitigation**: Manual testing + future Dash testing integration.

### 3. Device-Specific Tests Missing

No tests for CUDA/MPS device handling.

**Mitigation**: Device logic is simple passthrough, tested manually.

## Integration Testing (Separate)

For end-to-end testing with real models:

```bash
# 1. Generate small test dataset
python generate_dataset.py \
  --model imagenet \
  --checkpoint_path path/to/checkpoint.pth \
  --num_samples 10 \
  --layers encoder_bottleneck,midblock

# 2. Process embeddings
python process_embeddings.py \
  --model imagenet \
  --n_neighbors 5

# 3. Launch visualizer
python visualization_app.py \
  --embeddings data/embeddings/imagenet_umap_n5_d0.1.csv \
  --checkpoint_path path/to/checkpoint.pth

# 4. Manual test:
#    - Select point
#    - Find neighbors
#    - Generate image
#    - Verify green star appears
```

## Debugging Failed Tests

### Verbose Output

```bash
pytest test_activation_masking.py -v -s
```

`-s` shows print statements.

### Specific Failure

```bash
pytest test_activation_masking.py::TestActivationMask::test_masking_hook_replaces_output -vv
```

`-vv` shows full diff on assertion failures.

### Drop to Debugger

```bash
pytest test_activation_masking.py --pdb
```

### Keep Temp Files

Modify test to not use context manager:

```python
tmpdir = tempfile.mkdtemp()
print(f"Test files in: {tmpdir}")
# Don't delete, inspect manually
```

## Future Enhancements

1. **Dash Testing**: Add Selenium-based tests for UI
2. **Performance Tests**: Benchmark generation speed
3. **Regression Tests**: Compare generated images to baseline
4. **Property-based Tests**: Use Hypothesis for edge cases
5. **Integration Suite**: Full pipeline tests with real checkpoints
6. **SDXL Tests**: Once SDXL support added

## Maintenance

Run tests before:
- ✅ Committing changes
- ✅ Creating pull requests
- ✅ Releasing new versions

Update tests when:
- ✅ Adding new features
- ✅ Fixing bugs
- ✅ Refactoring code

## Questions?

See test files for detailed examples. Each test has clear docstrings explaining what it tests and why.
