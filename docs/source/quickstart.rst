Quick Start Guide
=================

This guide demonstrates basic usage of BowtieQGT for computing quantum geometric tensors and gradients.

Basic Usage
-----------

Computing QGT and Gradients
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

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
       ("XXII", -0.5),
       ("YYII", -0.5),
   ])

   # Initialize BowtieQGT
   bowtie = BowtieQGT(qc, obs, phase_fix=True, pbar=1)

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
   print(f"Gradient shape: {gradient.shape}")
   print(f"Variance: {variance}")

Real vs Imaginary Time Evolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For **real-time evolution** (VarQITE):

.. code-block:: python

   bowtie = BowtieQGT(qc, obs, VarQITE_gradient=True)

For **imaginary-time evolution**:

.. code-block:: python

   bowtie = BowtieQGT(qc, obs, VarQITE_gradient=False)

GPU Acceleration
~~~~~~~~~~~~~~~~

Enable GPU acceleration for faster computation:

.. code-block:: python

   bowtie = BowtieQGT(qc, obs, accelerator="GPU")

Phase Fixing
~~~~~~~~~~~~

Phase fixing improves numerical stability by removing contributions from the |0...0⟩ state:

.. code-block:: python

   # With phase fixing (recommended)
   bowtie = BowtieQGT(qc, obs, phase_fix=True)
   
   # Without phase fixing
   bowtie = BowtieQGT(qc, obs, phase_fix=False)

Progress Bars
~~~~~~~~~~~~~

Control verbosity with the ``pbar`` parameter:

.. code-block:: python

   # No progress bars
   bowtie = BowtieQGT(qc, obs, pbar=0)
   
   # Main computation progress
   bowtie = BowtieQGT(qc, obs, pbar=1)
   
   # Detailed progress including QGT computation
   bowtie = BowtieQGT(qc, obs, pbar=2)

Understanding the Output
------------------------

Generalized QGT Matrix Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The generalized QGT matrix has the following block structure:

.. code-block:: text

   ┌─────────────┬──────────────┐
   │     QGT     │   Gradient   │
   │  (N×N)      │   (N×M)      │
   ├─────────────┼──────────────┤
   │  Gradient†  │   Variance   │
   │  (M×N)      │   (M×M)      │
   └─────────────┴──────────────┘

Where:
- N = number of circuit parameters
- M = number of observable terms

QGT Block
~~~~~~~~~

The QGT (upper-left N×N block) is the metric tensor for the parameter space:

.. math::

   G_{ij} = \langle\partial_i\psi|\partial_j\psi\rangle - \langle\partial_i\psi|\psi\rangle\langle\psi|\partial_j\psi\rangle

Gradient Block
~~~~~~~~~~~~~~

The gradient (upper-right N×M block) contains energy derivatives:

.. math::

 \frac{\partial E }{\partial\theta_i} = \sum_{j=0}^{M-1} c_j \frac{\partial h_j}{\partial\theta_i} = \langle\partial_i\psi|h_j|\psi\rangle - \langle\partial_i\psi|\psi\rangle\langle\psi|h_j|\psi\rangle 

for a Hamiltonian:

.. math::

   H=\sum_{j=0}^{M-1} c_j h_j

Variance Block
~~~~~~~~~~~~~~

The variance (lower-right M×M block) contains observable variance information.


.. math::

    \text{Var}[H] = \sum_{i=0}^{M-1}\sum_{j=0}^{M-1} \langle\psi|h_i h_j|\psi\rangle - \langle\psi|h_i|\psi\rangle\langle\psi|h_j|\psi\rangle 



Next Steps
----------

- Explore the :doc:`api` for detailed API documentation
- Check :doc:`examples` for advanced usage patterns
- Review the test suite in ``test/test_bowtie_qgt.py`` for more examples
