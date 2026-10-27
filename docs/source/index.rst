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
   changelog

Key Features
------------

- **Sparse tensor operations** for efficient overlap computation
- **Parallel statevector simulation** using Qiskit Aer
- **Automatic identification** of active qubits per parameter/observable
- **McLachlan gradients** for real (VarQRTE) and imaginary (VarQITE) time evolution
- **Any Aer statevector simulator**, including GPU

Quick Example
-------------

.. code-block:: python

   from qiskit import QuantumCircuit
   from qiskit.quantum_info import SparsePauliOp
   from bowtie_qgt.bowtieqgt import BowtieQGT

   # Create a parameterized circuit
   qc = QuantumCircuit(4)
   # ... add parameterized gates ...

   hamiltonian = SparsePauliOp.from_list([("ZIII", 1.0), ("IZII", 1.0)])
   bowtie = BowtieQGT(qc, hamiltonian)

   generalized_qgt, energy = bowtie.evaluate([0.1] * qc.num_parameters)
   qgt = bowtie.extract_qgt(generalized_qgt)
   b = bowtie.extract_gradient(generalized_qgt, which="real_time")

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
