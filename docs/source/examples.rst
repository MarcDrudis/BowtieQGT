Examples
========

Real- and Imaginary-Time Evolution with McLachlan's Principle
-------------------------------------------------------------

This example runs two separate evolutions of an 8-qubit transverse-field Ising chain

.. math::

   H = -J \sum_i Z_i Z_{i+1} - h \sum_i X_i .

For :math:`h < J` the ground state is ferromagnetically ordered along :math:`Z`, for
:math:`h > J` it is a paramagnet polarized along :math:`X`, and :math:`h = J` is the
quantum critical point.

.. list-table::
   :header-rows: 1
   :stub-columns: 1

   * -
     - Real time (VarQRTE)
     - Imaginary time (VarQITE)
   * - Evolution
     - :math:`|\psi(t)\rangle = e^{-iHt}|\psi_0\rangle`
     - :math:`|\psi(\tau)\rangle \propto e^{-H\tau}|\psi_0\rangle`
   * - What it is for
     - Dynamics: what an experiment would see after a quench
     - Cooling: excited states decay as :math:`e^{-(E_k-E_0)\tau}`, leaving the ground state
   * - ``extract_gradient``
     - ``which="real_time"``
     - ``which="imag_time"``
   * - In this example
     - :math:`J = 0.25,\ h = 1`, start :math:`|0\ldots0\rangle`, plot :math:`m_z(t)`
     - :math:`J = h = 1`, start :math:`|+\ldots+\rangle`, plot :math:`E(\tau)`, :math:`\mathrm{Var}[H](\tau)`

In both cases, McLachlan's variational principle turns the evolution into the linear
system :math:`\mathrm{Re}(G)\,\dot\theta = b` at every step. Only :math:`b` depends on the
direction. ``BowtieQGT`` provides :math:`G` and :math:`b`, and ``scipy.integrate.solve_ivp``
integrates :math:`\dot\theta`. A small regularization :math:`\lambda I` takes care of the
redundant directions of :math:`G`.

1. Real time: magnetization after a quench
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Starting from :math:`|0\ldots0\rangle` (all spins along :math:`+Z`), the strong field
makes every spin precess about :math:`X`, so the magnetization
:math:`m_z = \langle\sum_i Z_i\rangle / n` swings from :math:`+1` down to about
:math:`-0.9` and back, with period :math:`\sim\pi/h`. The weak :math:`ZZ` coupling
spreads correlations along the chain and dephases the precession, so the oscillations
shrink and lose their regularity. VarQRTE follows the exact curve over the whole window
(left panel).

2. Imaginary time: ground state at the critical point
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Starting from :math:`|+\ldots+\rangle`, the ground state at :math:`h \to \infty`, the energy
decreases monotonically to the exact critical ground energy (middle panel). The variance
:math:`\mathrm{Var}[H]` falls by six orders of magnitude (right panel), which certifies that
the state is an eigenstate and not just a low-energy state. Imaginary time :math:`\tau` has
no physical meaning here; it only measures how far the cooling has progressed.

Choice of ansatz
~~~~~~~~~~~~~~~~

``n_local`` with ``rx``/``ry`` rotations and ``rzz`` bricks is a
Trotter circuit with free angles. It contains :math:`e^{-iHt}` at short times, and its
light cones stay narrow, which is what makes the bowtie method cheap. Hardware-efficient
ansätze with fixed CNOT chains, such as ``efficient_su2``, spread every light cone over the
whole chain and track this quench much less accurately (error above 0.05 from
:math:`t \approx 3` with 3 repetitions). With their default ``ry``/``rz`` gates they cannot
start the quench at all: at :math:`\theta = 0` the required direction
:math:`-iX|0\rangle` is outside the ansatz.

.. image:: _static/mclachlan_ising.png
   :alt: Left: real-time magnetization after a quench, VarQRTE vs exact. Middle and right: imaginary-time energy and variance of VarQITE.

The script runs in about seven minutes on a laptop. It needs ``matplotlib`` and ``scipy``. It is
also in the repository as ``docs/examples/mclachlan_ising.py``.

.. literalinclude:: ../examples/mclachlan_ising.py
   :language: python
   :start-at: import matplotlib

See Also
--------

- :doc:`quickstart`: conventions and the structure of the generalized QGT
- :doc:`api`: complete API reference
- ``test/test_bowtie_qgt.py``: checks against Qiskit's reference gradients on many circuits
