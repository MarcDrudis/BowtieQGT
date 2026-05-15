BowtieQGT Documentation
=======================

**BowtieQGT** is a Python library for efficient computation of Quantum Geometric Tensors (QGT), energy gradients, and variance for parameterized quantum circuits using the "bowtie" method.

The bowtie approach leverages light-cone structures to reduce computational overhead by focusing only on relevant qubits for each parameter and observable term, making it particularly efficient for large quantum circuits.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api
   examples

Key Features
------------

- **Sparse tensor operations** for efficient overlap computation
- **Parallel statevector simulation** using Qiskit Aer
- **Automatic identification** of active qubits per parameter/observable
- **Phase fixing** for improved numerical stability
- **GPU acceleration** support via Qiskit Aer

Quick Example
-------------

.. code-block:: python

   from qiskit import QuantumCircuit
   from qiskit.quantum_info import SparsePauliOp
   from bowtie_qgt.bowtieqgt import BowtieQGT

   # Create a parameterized circuit
   qc = QuantumCircuit(4)
   # ... add parameterized gates ...

   # Define an observable
   obs = SparsePauliOp.from_list([("ZIII", 1.0), ("IZII", 1.0)])

   # Initialize BowtieQGT
   bowtie = BowtieQGT(qc, obs, phase_fix=True)

   # Compute QGT and energy at parameter values
   params = {p: 0.1 for p in qc.parameters}
   gen_qgt, energy = bowtie.get_derivatives(params)

   # Extract QGT and gradient
   qgt = bowtie.extract_qgt(gen_qgt)
   gradient = bowtie.extract_gradient(gen_qgt)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
