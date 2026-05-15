Examples
========

This page provides practical examples demonstrating various use cases of BowtieQGT.

Example 1: Basic VQE Gradient Computation
------------------------------------------

Computing gradients for Variational Quantum Eigensolver (VQE):

.. code-block:: python

   from qiskit import QuantumCircuit
   from qiskit.circuit import ParameterVector
   from qiskit.quantum_info import SparsePauliOp
   from bowtie_qgt.bowtieqgt import BowtieQGT
   import numpy as np

   # Create a 4-qubit ansatz
   qc = QuantumCircuit(4)
   params = ParameterVector('θ', 12)
   
   idx = 0
   for layer in range(2):
       for q in range(4):
           qc.ry(params[idx], q)
           idx += 1
       for q in range(0, 3, 2):
           qc.cx(q, q + 1)
   
   for q in range(4):
       qc.ry(params[idx], q)
       idx += 1

   # Define Hamiltonian
   obs = SparsePauliOp.from_list([
       ("ZIII", -1.0), ("IZII", -1.0),
       ("IIZI", -1.0), ("IIIZ", -1.0),
       ("XXII", -0.5), ("IIYY", -0.5),
   ])

   # Initialize and compute
   bowtie = BowtieQGT(qc, obs, phase_fix=True)
   param_values = {p: np.random.random() for p in qc.parameters}
   gen_qgt, energy = bowtie.get_derivatives(param_values)
   gradient = bowtie.extract_gradient(gen_qgt)

   print(f"Energy: {energy.real:.6f}")
   print(f"Gradient norm: {np.linalg.norm(gradient):.6f}")


See Also
--------

- :doc:`api` - Complete API reference
- :doc:`quickstart` - Basic usage guide
- ``test/test_bowtie_qgt.py`` - Comprehensive test suite with more examples
