Quick Start Guide
=================

This guide shows how to compute the QGT, McLachlan gradients and the energy variance of a
parameterized circuit.

Basic Usage
-----------

.. code-block:: python

   from qiskit import QuantumCircuit
   from qiskit.circuit import ParameterVector
   from qiskit.quantum_info import SparsePauliOp
   from bowtie_qgt.bowtieqgt import BowtieQGT

   num_qubits = 4
   qc = QuantumCircuit(num_qubits)
   params = ParameterVector("θ", 8)
   for i in range(num_qubits):
       qc.ry(params[i], i)
   for i in range(num_qubits - 1):
       qc.cx(i, i + 1)
   for i in range(num_qubits):
       qc.ry(params[i + 4], i)

   hamiltonian = SparsePauliOp.from_list(
       [("ZIII", -1.0), ("IZII", -1.0), ("IIZI", -1.0), ("IIIZ", -1.0), ("XXII", -0.5)]
   )

   bowtie = BowtieQGT(qc, hamiltonian)
   generalized_qgt, energy = bowtie.evaluate([0.1] * qc.num_parameters)

   qgt = bowtie.extract_qgt(generalized_qgt)  # real part by default
   b_real = bowtie.extract_gradient(generalized_qgt, which="real_time")
   b_imag = bowtie.extract_gradient(generalized_qgt, which="imag_time")
   variance = bowtie.extract_variance(generalized_qgt)

``evaluate`` accepts either a sequence of values ordered like ``qc.parameters`` or a
``{Parameter: value}`` mapping. Every parameter must be bound.

Supported Circuits
------------------

Every parameter must be the bare angle of exactly one ``rx``, ``ry``, ``rz``, ``rxx``,
``ryy`` or ``rzz`` gate. Any non-parameterized gates are allowed. The constructor raises a
``ValueError`` for parameter expressions (``2*θ``), other parameterized gates, a parameter
that appears in several gates, classical bits, control flow, or a parameterized global
phase.

Identity terms in the Hamiltonian are allowed: they shift the energy and nothing else.

Real and Imaginary Time
-----------------------

The same ``BowtieQGT`` object serves both. You only choose the direction when you extract
the gradient:

.. list-table::
   :header-rows: 1
   :stub-columns: 1

   * -
     - Real time (VarQRTE)
     - Imaginary time (VarQITE)
   * - Target
     - :math:`|\psi(t)\rangle = e^{-iHt}|\psi_0\rangle`
     - :math:`|\psi(\tau)\rangle \propto e^{-H\tau}|\psi_0\rangle`
   * - Use
     - Quantum dynamics
     - Ground-state preparation
   * - Call
     - ``extract_gradient(g, which="real_time")``
     - ``extract_gradient(g, which="imag_time")``
   * - :math:`b_i`
     - :math:`\mathrm{Im}(\langle\partial_i\psi|H|\psi\rangle - \langle\partial_i\psi|\psi\rangle E)`
     - :math:`-\mathrm{Re}\langle\partial_i\psi|H|\psi\rangle = -\tfrac12\partial_i E`

McLachlan's variational principle gives :math:`\mathrm{Re}(G)\,\dot\theta = b` in both
cases. The QGT is the same; only :math:`b` changes:

.. code-block:: python

   qgt = bowtie.extract_qgt(generalized_qgt)

   # Real time: follow e^{-iHt}
   b = bowtie.extract_gradient(generalized_qgt, which="real_time")
   theta_dot = np.linalg.solve(qgt + 1e-6 * np.eye(len(b)), b)

   # Imaginary time: flow towards the ground state
   b = bowtie.extract_gradient(generalized_qgt, which="imag_time")
   theta_dot = np.linalg.solve(qgt + 1e-6 * np.eye(len(b)), b)

The names refer to the evolution, not to the part of the matrix that is taken: the
real-time gradient happens to be the real part of the mixed block of the generalized QGT,
but in terms of the raw overlap :math:`\langle\partial_i\psi|H|\psi\rangle` it is the
imaginary part. ``which=`` hides this bookkeeping. :doc:`examples` has a full run of both.

Choosing the Simulator
----------------------

``BowtieQGT`` runs the bowtie circuits on ``AerSimulator(method="statevector")`` by
default. Pass your own noiseless statevector simulator to choose the device, precision
or number of threads:

.. code-block:: python

   from qiskit_aer import AerSimulator

   gpu = AerSimulator(method="statevector", device="GPU")
   bowtie = BowtieQGT(qc, hamiltonian, simulator=gpu)

   eight_threads = AerSimulator(method="statevector", max_parallel_threads=8)
   bowtie = BowtieQGT(qc, hamiltonian, simulator=eight_threads)

Progress and Timings
--------------------

``verbosity=1`` shows progress bars while building the bowties and prints their light-cone
widths. ``verbosity=2`` also reports every ``evaluate`` call. The time spent in the last
call is always available:

.. code-block:: python

   bowtie.evaluate(values)
   bowtie.last_timings  # {"simulation": seconds, "assembly": seconds}

Understanding the Output
------------------------

The generalized QGT is the connected QGT of the circuit extended by one first-order
Trotter step :math:`e^{-i c_j h_j t}` per Hamiltonian term. It is complex Hermitian, with
``N`` circuit parameters followed by ``M`` non-identity Hamiltonian terms:

.. code-block:: text

   ┌─────────────┬──────────────┐
   │     QGT     │    mixed     │
   │   (N×N)     │    (N×M)     │
   ├─────────────┼──────────────┤
   │   mixed†    │ Hamiltonian  │
   │   (M×N)     │    (M×M)     │
   └─────────────┴──────────────┘

The parameter tangents are :math:`|\partial_i\psi\rangle` and the Hamiltonian tangents are
:math:`-i h_j|\psi\rangle`. For :math:`H = \sum_j c_j h_j`:

.. math::

   G_{ij} &= \langle\partial_i\psi|\partial_j\psi\rangle
            - \langle\partial_i\psi|\psi\rangle\langle\psi|\partial_j\psi\rangle \\
   b^{\text{real}}_i + i\, b^{\text{imag}}_i &= \sum_j \tilde G_{i, N+j}\, c_j \\
   \mathrm{Var}[H] &= \sum_{jk} c_j^*\, \tilde G_{N+j, N+k}\, c_k

with :math:`b^{\text{real}}_i = \mathrm{Im}(\langle\partial_i\psi|H|\psi\rangle
- \langle\partial_i\psi|\psi\rangle E)` and
:math:`b^{\text{imag}}_i = -\mathrm{Re}(\langle\partial_i\psi|H|\psi\rangle
- \langle\partial_i\psi|\psi\rangle E)`. For Hermitian :math:`H` the last term is
real and drops out, so :math:`b^{\text{imag}} = -\tfrac12\nabla_\theta E`.

``extract_qgt`` returns :math:`\mathrm{Re}\,G` by default, the Fubini-Study metric that
McLachlan's principle uses. ``part="imag"`` gives the Berry curvature and
``part="complex"`` the full QGT, for other variational principles. Rotations follow
Qiskit's convention :math:`e^{-i\theta P/2}`, so ``G`` matches
``qiskit_algorithms.gradients.ReverseQGT``.

Next Steps
----------

- :doc:`examples`: VarQRTE and VarQITE of an Ising chain
- :doc:`api`: full API reference
- :doc:`changelog`: release notes and migration guide
