# BowtieQGT

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20210386.svg)](https://doi.org/10.5281/zenodo.20210386)

**Efficient Quantum Geometric Tensor computation for parameterized quantum circuits**

BowtieQGT is a Python library for computing Quantum Geometric Tensors (QGT), energy gradients, and variance for parameterized quantum circuits using the "bowtie" method. The bowtie approach leverages light-cone structures to reduce computational overhead by focusing only on relevant qubits for each parameter and observable term, making it particularly efficient for large quantum circuits.

## Key Features

- **QGT computation via the Bowtie method** for efficient QGT computation of large quantum circuits
- **Sparse tensor operations** for efficient overlap computation
- **Parallel statevector simulation** using Qiskit Aer
- **GPU acceleration** support via Qiskit Aer
- **Support for VarQTE** (real and imaginary time evolution)

## Quick Start

### Installation

Install BowtieQGT using pip:

```bash
pip install git+https://github.com/MarcDrudis/BowtieQGT.git
```

For GPU support, development installation, and detailed setup instructions, see the [Installation Guide](https://MarcDrudis.github.io/BowtieQGT/installation.html).

### Basic Usage

```python
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from bowtie_qgt.bowtieqgt import BowtieQGT

# Create a parameterized quantum circuit
num_qubits = 4
qc = QuantumCircuit(num_qubits)
params = ParameterVector('θ', 8)

# Build a simple ansatz
for i in range(num_qubits):
    qc.ry(params[i], i)

for i in range(num_qubits - 1):
    qc.cx(i, i + 1)

for i in range(num_qubits):
    qc.ry(params[i + 4], i)

# Define a Hamiltonian observable
obs = SparsePauliOp.from_list([
    ("ZIII", -1.0),
    ("IZII", -1.0),
    ("IIZI", -1.0),
    ("IIIZ", -1.0),
])

# Initialize BowtieQGT with phase fixing
bowtie = BowtieQGT(qc, obs, phase_fix=True)

# Set parameter values
param_values = {p: 0.1 for p in qc.parameters}

# Compute generalized QGT and energy
gen_qgt, energy = bowtie.get_derivatives(param_values)

# Extract components
qgt = bowtie.extract_qgt(gen_qgt)
gradient = bowtie.extract_gradient(gen_qgt)
variance = bowtie.extract_variance(gen_qgt, energy)

print(f"Energy: {energy}")
print(f"QGT shape: {qgt.shape}")
print(f"Gradient: {gradient}")
```

## What is the Quantum Geometric Tensor?

The Quantum Geometric Tensor (QGT) is a fundamental object in quantum information that characterizes the geometry of the parameter space of a quantum state. For a parameterized quantum state |ψ(θ)⟩, the QGT is defined as:

```
G_ij = Re[⟨∂_i ψ|∂_j ψ⟩ - ⟨∂_i ψ|ψ⟩⟨ψ|∂_j ψ⟩]
```

The QGT is essential for:

- **Variational Quantum Algorithms**: Natural gradient descent and quantum natural gradient optimization
- **VarQITE**: Variational Quantum Imaginary Time Evolution
- **Quantum Fisher Information**: Understanding parameter estimation precision
- **Quantum Speed Limits**: Characterizing the rate of quantum evolution

## The Bowtie Method

The bowtie method exploits the locality structure of quantum circuits to efficiently compute the QGT. Key insights:

1. **Light-cone structure**: Each parameter only affects a subset of qubits (its "light cone")
2. **Observable locality**: Each term in the observable only acts on specific qubits
3. **Sparse overlaps**: Only compute overlaps between states that share active qubits

This reduces the computational complexity from exponential in the total number of qubits to exponential in the maximum light-cone size, which is typically much smaller.

## Advanced Features

### GPU Acceleration

Enable GPU acceleration for faster computation on large circuits:

```python
bowtie = BowtieQGT(qc, obs, accelerator="GPU")
```

See the [Installation Guide](https://MarcDrudis.github.io/BowtieQGT/installation.html) for GPU setup instructions.

### VarQITE Support

For real-time evolution (VarQITE):

```python
bowtie = BowtieQGT(qc, obs, VarQITE_gradient=True)
```

For imaginary-time evolution:

```python
bowtie = BowtieQGT(qc, obs, VarQITE_gradient=False)
```

### Phase Fixing

Phase fixing improves numerical stability by removing contributions from the |0...0⟩ state:

```python
bowtie = BowtieQGT(qc, obs, phase_fix=True)  # Recommended
```

## Documentation

📚 **[Read the full documentation online](https://MarcDrudis.github.io/BowtieQGT/)**

The documentation includes:

- [Installation Guide](https://MarcDrudis.github.io/BowtieQGT/installation.html) - Setup instructions for users and developers
- [Quick Start Guide](https://MarcDrudis.github.io/BowtieQGT/quickstart.html) - Basic usage examples
- [API Reference](https://MarcDrudis.github.io/BowtieQGT/api.html) - Detailed API documentation
- [Examples](https://MarcDrudis.github.io/BowtieQGT/examples.html) - Advanced usage patterns

You can also build the documentation locally from the `docs/` directory using Sphinx.

## Development

### Prerequisites

This project uses [uv](https://docs.astral.sh/uv/) for fast, reliable Python package management.

### Development Installation

```bash
# Clone the repository
git clone https://github.com/MarcDrudis/BowtieQGT.git
cd BowtieQGT

# Create virtual environment and install in editable mode
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"

# Install pre-commit hooks
uv run pre-commit install
```

### Running Tests

```bash
uv run pytest
```

### Code Quality

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
uv run ruff check .        # Check for issues
uv run ruff check . --fix  # Auto-fix issues
```

Pre-commit hooks automatically run these checks before each commit.

## Requirements

- Python 3.11+
- Qiskit 2.0+
- Qiskit Aer
- NumPy 2.0+
- Additional dependencies: matplotlib, tqdm, qiskit-algorithms

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Citation

If you use BowtieQGT in your research, please cite:

Drudis, M. (2026). BowtieQGT: Efficient Quantum Geometric Tensor Computation (Version 1.0.0) [Computer software]. https://doi.org/10.5281/zenodo.20210386

Or in BibTeX format:

```bibtex
@software{drudis2026bowtieqgt,
  author = {Drudis, Marc},
  title = {BowtieQGT: Efficient Quantum Geometric Tensor Computation},
  year = {2026},
  version = {1.0.0},
  doi = {10.5281/zenodo.20210386},
  url = {https://github.com/MarcDrudis/BowtieQGT}
}
```

## Contributing

For contributions, please follow the development setup above and submit pull requests through the GitHub repository.

## Support

For issues, questions, or feature requests, please use the [GitHub Issues](https://github.com/MarcDrudis/BowtieQGT/issues) page.
