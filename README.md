# Wavelet-Coupled QMC for Rough Volatility

This project implements a wavelet-coupled Quasi-Monte Carlo (QMC) method for simulating rough fractional Brownian motion (fBM) paths, with a focus on rough volatility modeling in finance.

## Key Features

1. **Wavelet-coupled Sobol sequences** for efficient rough fBM path generation
2. **Baseline comparisons** against Cholesky decomposition and Euler-Maruyama
3. **Numba-accelerated** critical paths for performance
4. **Modular design** for easy extension and testing
5. **Comprehensive testing** with statistical validation

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── core/
│   ├── wavelet_qmc.py      # Wavelet-QMC path generation
│   ├── baselines.py        # Traditional methods (Cholesky, MC)
│   └── wavelet_utils.py    # Wavelet-specific math utilities
├── configs/
│   └── rough_vol_params.py # Configuration parameters
├── tests/
│   ├── benchmark_rough_vol.py  # Statistical tests
│   └── visualization.py    # Path visualization
└── README.md
```

## Usage

### Basic Path Generation

```python
from core.wavelet_qmc import rough_fbm_paths
from configs.rough_vol_params import ROUGH_VOL_CONFIG

# Generate paths using default config
paths = rough_fbm_paths(**ROUGH_VOL_CONFIG)
```

### Running Tests

```bash
# Run all tests
pytest tests/benchmark_rough_vol.py -v

# Run with performance benchmarks
pytest tests/benchmark_rough_vol.py -v --benchmark-only
```

### Visualization

```bash
# Generate path visualizations
python tests/visualization.py --H 0.1 --j_max 6
```

## Performance

The wavelet-QMC method offers significant advantages:

1. **Speed**: Faster than Cholesky decomposition for large path counts
2. **Accuracy**: Better than Euler-Maruyama for rough volatility (H < 0.5)
3. **Memory**: More memory-efficient than Cholesky for large time steps
