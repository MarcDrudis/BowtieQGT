Installation
============

This guide covers installation options for BowtieQGT, from basic installation for end users to advanced development setups.

Basic Installation
------------------

For most users who simply want to use BowtieQGT, the simplest installation method is using pip:

.. code-block:: bash

   pip install git+https://github.com/MarcDrudis/BowtieQGT.git


This will install BowtieQGT and all required dependencies, including:

- NumPy (≥2.0)
- Qiskit (≥2.0)
- Qiskit Algorithms (≥0.4.0)
- Qiskit Aer (for simulation)
- Matplotlib (for visualization)
- Additional utilities (tqdm, icecream, ddt)

After installation, you can verify it works by importing the package:

.. code-block:: python

   from bowtie_qgt.bowtieqgt import BowtieQGT
   print("BowtieQGT installed successfully!")

GPU Acceleration with Qiskit Aer
---------------------------------

BowtieQGT supports GPU acceleration through Qiskit Aer's GPU-enabled statevector simulator. This can significantly speed up computations for large quantum circuits.

Prerequisites
~~~~~~~~~~~~~

To use GPU acceleration, you need:

1. **CUDA-capable GPU**: An NVIDIA GPU with CUDA support
2. **CUDA Toolkit**: Version 11.2 or later
3. **GPU-enabled Qiskit Aer**: A special build of Qiskit Aer with GPU support

Installing GPU-Enabled Qiskit Aer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Qiskit Aer provides GPU support for accelerated simulations on Linux systems with NVIDIA GPUs.

**Prerequisites**: CUDA® 10.1 or newer must be installed, along with appropriate NVIDIA GPU drivers. Follow the CUDA® installation procedure from `NVIDIA's website <https://www.nvidia.com/drivers>`_.

**For x86_64 Linux systems**, install the pre-built GPU package:

.. code-block:: bash

   pip install qiskit-aer-gpu

This will replace your current ``qiskit-aer`` installation with a GPU-enabled version that provides the same functionality plus GPU support for statevector, density matrix, and unitary simulators.

.. note::
   The ``qiskit-aer-gpu`` package is only available for x86_64 Linux. For other platforms with CUDA support, you must build from source.

**Building from source with GPU support**: For detailed instructions on building Qiskit Aer from source with GPU support, refer to the official documentation:

- `Qiskit Aer Installation Guide <https://qiskit.github.io/qiskit-aer/getting_started.html>`_
- `GPU-specific instructions <https://qiskit.github.io/qiskit-aer/howtos/running_gpu.html>`_

**Verify GPU support**:

.. code-block:: python

   from qiskit_aer import AerSimulator
   
   # Check available devices
   sim = AerSimulator(method='statevector')
   print(sim.available_devices())
   # Should show ['CPU', 'GPU'] if GPU support is enabled

Using GPU Acceleration
~~~~~~~~~~~~~~~~~~~~~~

To use GPU acceleration with BowtieQGT, configure the Aer backend when initializing:

.. code-block:: python

   from qiskit_aer import AerSimulator
   from bowtie_qgt.bowtieqgt import BowtieQGT
   
   # Create GPU-enabled simulator
   gpu_sim = AerSimulator(method='statevector', device='GPU')
   
   # Initialize BowtieQGT with GPU backend
   bowtie = BowtieQGT(
       circuit,
       observable,
       backend=gpu_sim,
       phase_fix=True
   )

.. note::
   GPU acceleration is most beneficial for circuits with 15+ qubits. For smaller circuits, CPU execution may be faster due to GPU overhead.

Performance Tips
~~~~~~~~~~~~~~~~

- **Batch size**: Adjust the number of parallel circuit evaluations based on GPU memory
- **Circuit depth**: Deeper circuits benefit more from GPU acceleration
- **Memory management**: Monitor GPU memory usage for very large circuits

Development Installation
-------------------------

For developers who want to contribute to BowtieQGT or modify the source code, we recommend using `uv <https://docs.astral.sh/uv/>`_. 


Installing uv
~~~~~~~~~~~~~

To install uv:

.. code-block:: bash

   # macOS and Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

   # Or via pip
   pip install uv

Cloning the Repository
~~~~~~~~~~~~~~~~~~~~~~

Clone the BowtieQGT repository:

.. code-block:: bash

   git clone https://github.com/MarcDrudis/BowtieQGT.git
   cd BowtieQGT

Development Install with uv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install BowtieQGT in editable mode with all development dependencies:

.. code-block:: bash

   # Create and activate a virtual environment
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

   # Install in editable mode with dev dependencies
   uv pip install -e ".[dev]"

Alternatively, use uv's sync command to install from the lock file:

.. code-block:: bash

   uv sync --all-extras

This installs:

- **Core dependencies**: All runtime requirements
- **Development tools**: pytest, ruff, pre-commit, towncrier
- **Documentation tools**: Sphinx, Furo theme, autodoc extensions

Running Without Activation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can run commands directly without activating the virtual environment:

.. code-block:: bash

   uv run pytest              # Run tests
   uv run ruff check .        # Run linter
   uv run python script.py    # Run Python scripts

Setting Up Pre-commit Hooks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install pre-commit hooks to automatically check code quality before commits:

.. code-block:: bash

   uv run pre-commit install

Now, every time you commit, the following checks will run automatically:

- Trailing whitespace removal
- End-of-file fixing
- YAML validation
- Ruff linting and formatting
- Changelog fragment validation

Verifying Your Installation
----------------------------

After installation, verify everything works correctly:

.. code-block:: bash

   # Run the test suite
   uv run pytest

   # Or if you've activated the virtual environment
   pytest

All tests should pass. If you encounter any issues, please check:

1. Python version is 3.11 or later: ``python --version``
2. All dependencies are installed: ``pip list``
3. For GPU issues, verify CUDA installation: ``nvidia-smi``

Troubleshooting
---------------

Common Installation Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Import errors after installation**:
   Ensure you're using the correct Python environment. Check with ``which python`` (Linux/macOS) or ``where python`` (Windows).

**GPU not detected**:
   Verify CUDA installation with ``nvidia-smi`` and ensure Qiskit Aer was built with GPU support.

**Permission errors during installation**:
   Use ``pip install --user`` or create a virtual environment to avoid system-wide installation.

**Build failures with GPU-enabled Aer**:
   Ensure you have the correct CUDA Toolkit version and all build dependencies installed.

Getting Help
~~~~~~~~~~~~

If you encounter issues not covered here:

- Check the `GitHub Issues <https://github.com/MarcDrudis/BowtieQGT/issues>`_
- Review the Qiskit Aer `GPU documentation <https://qiskit.github.io/qiskit-aer/howtos/running_gpu.html>`_

Next Steps
----------

After installation, proceed to the :doc:`quickstart` guide to learn how to use BowtieQGT for your quantum computing workflows.

