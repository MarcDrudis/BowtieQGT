# SPDX-License-Identifier: Apache-2.0
# (C) Copyright IBM 2025.

"""Bowtie Quantum Geometric Tensor (QGT) computation module.

This module computes the Quantum Geometric Tensor (QGT), McLachlan gradients and the
energy variance of a parameterized quantum circuit with the "bowtie" method. Each
parameter and each Hamiltonian term gets an auxiliary circuit (a bowtie) restricted
to its light cone, so the cost is exponential in the light-cone width rather than in
the total number of qubits.

All quantities come out of a single complex matrix, the *generalized QGT*: the
connected QGT of the circuit extended by one first-order Trotter step of the
Hamiltonian. See :class:`BowtieQGT` for the conventions.

Example:
    >>> bowtie = BowtieQGT(qc, hamiltonian)
    >>> generalized_qgt, energy = bowtie.evaluate(parameter_values)
    >>> qgt = bowtie.extract_qgt(generalized_qgt)
    >>> b = bowtie.extract_gradient(generalized_qgt, which="real_time")
    >>> theta_dot = np.linalg.solve(qgt + 1e-6 * np.eye(len(b)), b)
"""

import warnings
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import copy
from itertools import product
from time import perf_counter

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from tqdm import tqdm

from bowtie_qgt.bowtie_circuits import (
    observable_bowtie,
    parameter_bowtie,
    remove_idle_qubits,
    validate_circuit,
)


def zeroth(tensor: np.ndarray):
    """Returns tensor[0,0,0,0,...] of a given ndarray"""
    return tensor[(0,) * tensor.ndim]


def get_slice(AB: tuple[int], BC: tuple[int]):
    """Generate slice indices for projecting ψ_AB onto the overlap region with ψ_BC.

    Computes the slice to apply to statevector ψ_AB when computing its overlap
    with ψ_BC. The Hilbert space is decomposed into three regions:
    - Region A: qubits where only ψ_AB has support (projected to |0⟩)
    - Region B: qubits where both ψ_AB and ψ_BC have support (overlap computed)
    - Region C: qubits where only ψ_BC has support (projected to |0⟩)

    For qubits in region B (intersection of AB and BC), the slice uses `slice(None)`
    to keep all amplitudes. For qubits in region A (in AB but not in BC), the slice
    uses index 0 to project onto |0⟩.

    Args:
        AB: Tuple of active qubit indices for statevector ψ_AB.
        BC: Tuple of active qubit indices for statevector ψ_BC.

    Returns:
        List of slice objects or integers for indexing the ψ_AB tensor:
        - `slice(None)` for qubits in the overlap region B (qubits in both AB and BC)
        - `0` for qubits in region A (qubits in AB but not in BC)
        The list is in reversed order to match tensor dimension ordering.

    Note:
        When precomputing slices for overlap ⟨ψ_AB|ψ_BC⟩, both directions must be
        computed and stored: one slice for ψ_AB and one for ψ_BC, since the overlap
        is not symmetric when the active qubit sets differ.

    Example:
        >>> # ψ_AB has support on qubits (0, 1, 2), ψ_BC on qubits (1, 2, 3,4)
        >>> # Region A = {0}, Region B = {1, 2}, Region C = {3,4}
        >>> get_slice((0, 1, 2), (1, 2, 3, 4))
        [slice(None, None, None), slice(None, None, None), 0]
        >>> # Qubit 0 (region A) → 0, qubits 1,2 (region B) → slice(None)
    """
    return [(slice(None) if (i in BC) else 0) for i in reversed(AB)]


def sparse_overlap_tensors(svAB: np.ndarray, sliceab, svBC: np.ndarray, slicebc):
    """Compute the overlap ⟨ψ_AB|ψ_BC⟩ between statevectors with different support.

    Calculates the inner product between two statevectors by projecting them onto
    their common support region (region B). The Hilbert space is decomposed as:
    - Region A: qubits where only ψ_AB has support → projected to |0⟩ in ψ_AB
    - Region B: qubits where both have support → full overlap computed
    - Region C: qubits where only ψ_BC has support → projected to |0⟩ in ψ_BC

    The slices `sliceab` and `slicebc` project each statevector onto region B,
    setting amplitudes in regions A and C to their |0⟩ components.

    Args:
        svAB: Statevector tensor ψ_AB reshaped to [2, 2, ..., 2] with dimensions
            corresponding to its active qubits.
        sliceab: Slice for projecting ψ_AB onto region B.
        svBC: Statevector tensor ψ_BC reshaped to [2, 2, ..., 2] with dimensions
            corresponding to its active qubits.
        slicebc: Slice for projecting ψ_BC onto region B.

    Returns:
        Complex number representing the overlap ⟨ψ_AB|ψ_BC⟩ computed over the
        common support region B.

    Note:
        This is the core operation for computing QGT matrix entries. The sparse
        tensor approach avoids explicitly constructing full statevectors on the
        entire Hilbert space, significantly reducing memory and computation.
    """
    ab = svAB[*sliceab]
    bc = svBC[*slicebc]
    return np.vdot(ab, bc)


def connected_overlap(svAB, sliceab, svBC, slicebc):
    """Compute the connected overlap ``⟨ψ_AB|ψ_BC⟩ - ⟨ψ_AB|0⟩⟨0|ψ_BC⟩``.

    Bowtie states are expressed in the frame where the variational state is ``|0...0⟩``,
    so subtracting the ``|0...0⟩`` components projects out the global-phase direction.
    This makes every block of the generalized QGT gauge invariant.

    Args:
        svAB: First statevector tensor.
        sliceab: Slice indices for svAB.
        svBC: Second statevector tensor.
        slicebc: Slice indices for svBC.

    Returns:
        Complex number representing the connected overlap.
    """
    return sparse_overlap_tensors(svAB, sliceab, svBC, slicebc) - np.conj(zeroth(svAB)) * zeroth(
        svBC
    )


class BowtieQGT:
    r"""Quantum Geometric Tensor, gradients and variance via light-cone bowties.

    Conventions:
        Let :math:`|\psi(\theta)\rangle = U(\theta)|0\rangle` and
        :math:`H = \sum_j c_j h_j` with Pauli terms :math:`h_j`. Extend the
        parameter tangents :math:`|\partial_i\psi\rangle` with one tangent per
        Hamiltonian term, :math:`-i h_j|\psi\rangle` (the derivative of
        :math:`e^{-i c_j h_j t}` with respect to :math:`c_j t`, i.e. a first-order
        Trotter step appended to the circuit). The *generalized QGT* is the
        connected QGT of all these tangents :math:`\tau_a`:

        .. math::

            \tilde G_{ab} = \langle\tau_a|\tau_b\rangle
                - \langle\tau_a|\psi\rangle\langle\psi|\tau_b\rangle,

        a complex Hermitian matrix of shape ``(N + M, N + M)`` with ``N`` circuit
        parameters and ``M`` non-identity Hamiltonian terms, ordered like
        ``qc.parameters`` and then the terms of ``hamiltonian``. Its blocks give:

        * QGT: :math:`G = \tilde G_{:N,:N}`. Its real part is the Fubini-Study
          metric (the matrix of McLachlan's principle), its imaginary part the Berry
          curvature. Rotations follow Qiskit's convention :math:`e^{-i\theta P/2}`,
          so this matches ``qiskit_algorithms.gradients.ReverseQGT``.
        * McLachlan gradients :math:`\tilde G_{:N,N:}\,c`: its real part is the
          real-time vector :math:`b_i = \mathrm{Im}(\langle\partial_i\psi|H|\psi\rangle
          - \langle\partial_i\psi|\psi\rangle E)`, its imaginary part the
          imaginary-time vector :math:`b_i = -\mathrm{Re}(\langle\partial_i\psi|H|\psi\rangle
          - \langle\partial_i\psi|\psi\rangle E)`, which equals
          :math:`-\tfrac12\,\partial_i E` for Hermitian :math:`H`.
        * Variance: :math:`\mathrm{Var}[H] = c^\dagger \tilde G_{N:,N:}\, c`.

        McLachlan's principle then reads :math:`\mathrm{Re}(G)\,\dot\theta = b`.
        Identity terms of the Hamiltonian have no tangent; they only shift the energy.

    Attributes:
        qc: The parameterized quantum circuit.
        hamiltonian: The Hamiltonian passed at construction.
        coeffs: Coefficients of the non-identity Hamiltonian terms, in bowtie order.
        energy_offset: Sum of the coefficients of identity terms.
        simulator: Aer simulator that runs the bowtie circuits.
        verbosity: Progress reporting level, see :meth:`__init__`.
        bowties: Bowtie circuits, parameters first and then Hamiltonian terms.
        active_qubits: Qubits each bowtie acts on.
        non_zero_indices: ``(i, j)`` pairs, ``i >= j``, whose bowties share qubits.
            Every other entry of the generalized QGT is exactly zero.
        slices: Precomputed tensor slices for each pair in ``non_zero_indices``.
        last_timings: Seconds spent in the last :meth:`evaluate` call, with keys
            ``"simulation"`` and ``"assembly"``.
    """

    def __init__(
        self,
        qc: QuantumCircuit,
        hamiltonian: SparsePauliOp,
        simulator: AerSimulator | None = None,
        verbosity: int = 0,
    ):
        """Build and transpile the bowtie circuits.

        Args:
            qc: Parameterized circuit. Every parameter must be the bare angle of
                exactly one ``rx``, ``ry``, ``rz``, ``rxx``, ``ryy`` or ``rzz`` gate.
            hamiltonian: Hamiltonian as a ``SparsePauliOp``. Identity terms are
                allowed and only shift the energy.
            simulator: Noiseless statevector ``AerSimulator`` used to run the
                bowties. Defaults to ``AerSimulator(method="statevector")``. Pass your
                own to pick the device, precision or parallelism, e.g.
                ``AerSimulator(method="statevector", device="GPU")`` or
                ``AerSimulator(method="statevector", max_parallel_threads=8)``.
            verbosity: 0 is silent. 1 shows progress bars while building the
                bowties and prints the distribution of light-cone widths. 2 also
                reports progress and timings on every :meth:`evaluate` call.

        Raises:
            ValueError: If the circuit is not supported (see
                :func:`~bowtie_qgt.bowtie_circuits.validate_circuit`) or the simulator
                is not a noiseless statevector simulator.

        Warns:
            UserWarning: If the Hamiltonian is a multiple of the identity.
        """
        validate_circuit(qc)
        self.qc = qc
        self.hamiltonian = hamiltonian
        self.verbosity = verbosity
        self.simulator = simulator or AerSimulator(method="statevector")
        if self.simulator.options.method != "statevector" or self.simulator.options.noise_model:
            raise ValueError(
                "The simulator must be a noiseless AerSimulator with method='statevector'."
            )
        self.last_timings: dict[str, float] = {}
        show_progress = verbosity >= 1

        terms = hamiltonian.to_sparse_list()
        self.energy_offset = sum(coeff for term, _, coeff in terms if not term)
        terms = [(term, indices, coeff) for term, indices, coeff in terms if term]
        if not terms:
            warnings.warn(
                "Trivial Hamiltonian: it is a multiple of the identity, so its gradient "
                "and variance vanish and the bowtie method has nothing to compute.",
                stacklevel=2,
            )
        self.coeffs = np.array([coeff for _, _, coeff in terms], dtype=complex)

        dag = circuit_to_dag(qc)
        parameter_bowties = [
            remove_idle_qubits(parameter_bowtie(copy(dag), p))
            for p in tqdm(qc.parameters, "Parameter bowties", disable=not show_progress)
        ]
        hamiltonian_bowties = [
            remove_idle_qubits(observable_bowtie(qc, term, indices))
            for term, indices, _ in tqdm(terms, "Hamiltonian bowties", disable=not show_progress)
        ]
        if show_progress:
            for name, bowties in [
                ("Parameter", parameter_bowties),
                ("Hamiltonian", hamiltonian_bowties),
            ]:
                widths = Counter(len(active) for _, active in bowties)
                print(f"{name} bowtie widths (qubits: count): {dict(sorted(widths.items()))}")

        bowties = parameter_bowties + hamiltonian_bowties
        self.active_qubits = [active for _, active in bowties]
        for bw, _ in bowties:
            bw.save_statevector()
        # Lower any gates Aer cannot run natively (e.g. high-level library gates).
        self.bowties = [
            transpile(bw, self.simulator)
            for bw, _ in tqdm(bowties, "Transpiling bowties", disable=not show_progress)
        ]

        # Bowties on disjoint qubits have a vanishing connected overlap, so skip them.
        self.non_zero_indices = [
            (i, j)
            for i, j in product(range(len(self.active_qubits)), repeat=2)
            if i >= j and set(self.active_qubits[i]).intersection(self.active_qubits[j])
        ]
        self.slices = [
            (
                get_slice(self.active_qubits[i], self.active_qubits[j]),
                get_slice(self.active_qubits[j], self.active_qubits[i]),
            )
            for i, j in self.non_zero_indices
        ]
        # Parameter tangents are -i/2 times their bowtie state (Qiskit rotations are
        # exp(-i θ P / 2)); Hamiltonian tangents are -i times theirs. The phases cancel
        # in every overlap, leaving a real rescaling.
        self._scale = np.concatenate([np.full(qc.num_parameters, 0.5), np.ones(len(terms))])

    def evaluate(
        self, parameter_values: Mapping[Parameter, float] | Sequence[float]
    ) -> tuple[np.ndarray, float | complex]:
        r"""Compute the generalized QGT and the energy at the given parameters.

        Args:
            parameter_values: Values for every circuit parameter, either as a
                sequence ordered like ``qc.parameters`` or as any mapping that
                ``QuantumCircuit.assign_parameters`` accepts (``Parameter``,
                ``ParameterVector`` or parameter-name keys).

        Returns:
            ``(generalized_qgt, energy)``. ``generalized_qgt`` is the complex
            ``(N + M, N + M)`` matrix described in :class:`BowtieQGT`; pass it to the
            ``extract_*`` methods. ``energy`` is :math:`\langle\psi|H|\psi\rangle`,
            a float when the Hamiltonian is Hermitian (real coefficients).

        Raises:
            ValueError: If a circuit parameter has no value.
        """
        if not isinstance(parameter_values, Mapping):
            parameter_values = list(parameter_values)
            if len(parameter_values) != self.qc.num_parameters:
                raise ValueError(
                    f"Expected {self.qc.num_parameters} parameter values, "
                    f"got {len(parameter_values)}."
                )
        # Let Qiskit resolve the keys (Parameter, ParameterVector, names) once.
        missing = self.qc.assign_parameters(parameter_values, strict=False).parameters
        if missing:
            raise ValueError(f"Missing values for parameters: {[p.name for p in missing]}")
        if not isinstance(parameter_values, Mapping):
            parameter_values = dict(zip(self.qc.parameters, parameter_values))

        start = perf_counter()
        bound = [bw.assign_parameters(parameter_values, strict=False) for bw in self.bowties]
        results = self.simulator.run(bound).result().results if bound else []
        # Reshaping the statevectors allows for faster computation of sparse overlaps
        tensors = [
            np.reshape(result.data.statevector.data, [2] * len(active))
            for result, active in zip(results, self.active_qubits)
        ]
        simulated = perf_counter()

        generalized_qgt = np.zeros([len(self.bowties)] * 2, dtype=complex)
        pairs = tqdm(
            list(zip(self.non_zero_indices, self.slices)),
            "Generalized QGT",
            disable=self.verbosity < 2,
        )
        for (i, j), (slice_i, slice_j) in pairs:
            generalized_qgt[i, j] = connected_overlap(tensors[i], slice_i, tensors[j], slice_j)
        # Only the lower triangle was computed; fill the upper one by Hermiticity.
        generalized_qgt += np.tril(generalized_qgt, -1).conj().T
        generalized_qgt *= np.outer(self._scale, self._scale)

        n = self.qc.num_parameters
        energy = self.energy_offset + sum(zeroth(t) * c for t, c in zip(tensors[n:], self.coeffs))
        if np.allclose(self.coeffs.imag, 0) and np.isclose(np.imag(self.energy_offset), 0):
            energy = float(np.real(energy))
        self.last_timings = {
            "simulation": simulated - start,
            "assembly": perf_counter() - simulated,
        }
        if self.verbosity >= 2:
            print(
                f"Simulation {self.last_timings['simulation']:.3f}s, "
                f"assembly {self.last_timings['assembly']:.3f}s"
            )
        return generalized_qgt, energy

    def extract_qgt(self, generalized_qgt: np.ndarray, part: str = "real") -> np.ndarray:
        """Extract the ``(N, N)`` QGT block.

        Args:
            generalized_qgt: Matrix returned by :meth:`evaluate`.
            part: ``"real"`` (default) for the Fubini-Study metric, which is what
                McLachlan's variational principle uses for both real and imaginary
                time. ``"imag"`` for the Berry curvature, ``"complex"`` for the full
                QGT, as used by other variational principles.

        Returns:
            The requested part of the QGT.
        """
        qgt = generalized_qgt[: self.qc.num_parameters, : self.qc.num_parameters]
        return _part(qgt, part)

    def extract_gradient(self, generalized_qgt: np.ndarray, which: str) -> np.ndarray:
        r"""Extract McLachlan's right-hand side ``b``, so that ``Re(G) θ̇ = b``.

        Args:
            generalized_qgt: Matrix returned by :meth:`evaluate`.
            which: ``"real_time"`` for :math:`e^{-iHt}` (VarQRTE) or ``"imag_time"``
                for :math:`e^{-H\tau}` (VarQITE). For Hermitian ``H`` the
                imaginary-time vector is :math:`-\tfrac12\nabla_\theta E`.

        Returns:
            Real array of shape ``(N,)``.
        """
        n = self.qc.num_parameters
        mixed = generalized_qgt[:n, n:] @ self.coeffs
        if which == "real_time":
            return mixed.real
        if which == "imag_time":
            return mixed.imag
        raise ValueError(f"which must be 'real_time' or 'imag_time', got {which!r}")

    def extract_variance(self, generalized_qgt: np.ndarray) -> float:
        r"""Extract the energy variance :math:`\langle H^2\rangle - \langle H\rangle^2`.

        Args:
            generalized_qgt: Matrix returned by :meth:`evaluate`.

        Returns:
            The variance. It vanishes exactly on eigenstates of ``H``.
        """
        n = self.qc.num_parameters
        return float(np.real(self.coeffs.conj() @ generalized_qgt[n:, n:] @ self.coeffs))


def _part(matrix: np.ndarray, part: str) -> np.ndarray:
    if part == "real":
        return matrix.real
    if part == "imag":
        return matrix.imag
    if part == "complex":
        return matrix
    raise ValueError(f"part must be 'real', 'imag' or 'complex', got {part!r}")
