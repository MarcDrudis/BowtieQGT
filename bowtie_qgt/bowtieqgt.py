# SPDX-License-Identifier: Apache-2.0
# (C) Copyright IBM 2025.

"""Bowtie Quantum Geometric Tensor (QGT) computation module.

This module provides efficient computation of the Quantum Geometric Tensor (QGT),
energy gradients, and variance for parameterized quantum circuits using the
"bowtie" method. The bowtie approach leverages light-cone structures to reduce
computational overhead by focusing only on relevant qubits for each parameter
and observable term.

The main class, computes:
- Quantum Geometric Tensor (QGT) for variational quantum algorithms
- Energy gradients with respect to circuit parameters
- Observable variance (optional)
- Support for both real (VarQITE) and imaginary time evolution gradients

Key Features:
- Sparse tensor operations for efficient overlap computation
- Parallel statevector simulation using Qiskit Aer
- Automatic identification of active qubits per parameter/observable
- Phase fixing for improved numerical stability
- GPU acceleration support via Qiskit Aer

Example:
    >>> from qiskit import QuantumCircuit
    >>> from qiskit.quantum_info import SparsePauliOp
    >>> from bowtie_qgt.bowtieqgt import BowtieQGT
    >>>
    >>> # Create a parameterized circuit
    >>> qc = QuantumCircuit(4)
    >>> # ... add parameterized gates ...
    >>>
    >>> # Define an observable
    >>> obs = SparsePauliOp.from_list([("ZIII", 1.0), ("IZII", 1.0)])
    >>>
    >>> # Initialize BowtieQGT
    >>> bowtie = BowtieQGT(qc, obs, phase_fix=True)
    >>>
    >>> # Compute QGT and energy at parameter values
    >>> params = {p: 0.1 for p in qc.parameters}
    >>> gen_qgt, energy = bowtie.get_derivatives(params)
    >>>
    >>> # Extract QGT and gradient
    >>> qgt = bowtie.extract_qgt(gen_qgt)
    >>> gradient = bowtie.extract_gradient(gen_qgt)
"""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from itertools import product
from time import time

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from tqdm import tqdm

from bowtie_qgt.bowtie_circuits import observable_bowtie, parameter_bowtie, remove_idle_qwires


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


def tensor_phase_fix(svAB, sliceab, svBC, slicebc, phase_fix: bool):
    """Compute overlap with optional phase fixing correction.

    Calculates the overlap between two statevector tensors with an optional
    phase correction term. The phase fix removes the contribution from the
    |0...0⟩ computational basis state to improve numerical stability.

    Args:
        svAB: First statevector tensor.
        sliceab: Slice indices for svAB.
        svBC: Second statevector tensor.
        slicebc: Slice indices for svBC.
        phase_fix: Boolean or numeric value. If True, applies phase correction
            by subtracting the product of |0...0⟩ amplitudes.

    Returns:
        Complex number representing the phase-corrected overlap.
    """
    return (
        sparse_overlap_tensors(svAB, sliceab, svBC, slicebc)
        - zeroth(svAB) * zeroth(svBC) * phase_fix
    )


class BowtieQGT:
    """Efficient Quantum Geometric Tensor computation using the bowtie method.

    This class computes the generalized QGT matrix that includes:
    - QGT block: metric tensor for the parameter space
    - Gradient block: energy derivatives with respect to parameters
    - Variance block: observable variance (optional)

    The bowtie method constructs auxiliary circuits (bowties) for each parameter
    and observable term, focusing computation only on relevant qubits determined
    by light-cone analysis. This significantly reduces computational cost for
    large quantum circuits.

    Attributes:
        qc: The parameterized quantum circuit.
        obs: Observable as a SparsePauliOp for energy/gradient computation.
        phase_fix: Whether phase fixing is enabled.
        pbar: Progress bar verbosity level (0=off, 1=minimal, 2=detailed).
        accelerator: Device for simulation ("CPU" or "GPU").
        VarQITE_gradient: If True, computes real gradients (VarQITE style);
            if False, computes imaginary gradients.
        simulator: Qiskit Aer statevector simulator instance.
        bowties: List of all bowtie circuits (parameters + observables).
        active_qubits: Tuple of active qubit indices for each bowtie.
        non_zero_indices: List of (i,j) index pairs with non-zero overlaps.
        slices: Precomputed slice objects for efficient tensor indexing.

    Example:
        >>> qc = QuantumCircuit(4)
        >>> # ... build parameterized circuit ...
        >>> obs = SparsePauliOp.from_list([("ZIII", 1.0)])
        >>> bowtie = BowtieQGT(qc, obs, phase_fix=True, pbar=1)
        >>> params = {p: 0.5 for p in qc.parameters}
        >>> gen_qgt, energy = bowtie.get_derivatives(params)
    """

    def __init__(
        self,
        qc: QuantumCircuit,
        obs: SparsePauliOp,
        phase_fix: bool = True,
        pbar: int = 0,
        verbose_init: bool = False,
        accelerator: str = "CPU",
        compute_variance: bool = False,
        VarQITE_gradient: bool = False,
    ):
        """Initialize the BowtieQGT calculator.

        Args:
            qc: Parameterized quantum circuit to analyze.
            obs: Observable as SparsePauliOp for computing energy and gradients.
            phase_fix: If True, applies phase correction.
                Recommended for most cases. Default: True.
            pbar: Progress bar verbosity level:
                - 0: No progress bars
                - 1: Show main computation progress
                - 2: Show detailed progress including QGT computation
                Default: 0.
            verbose_init: If True, prints distribution statistics of bowtie
                circuit sizes during initialization. Default: False.
            accelerator: Simulation device, either "CPU" or "GPU". GPU requires
                appropriate Qiskit Aer GPU support. Default: "CPU".
            VarQITE_gradient: If True, computes real-time gradients (VarQITE);
                if False, computes imaginary-time gradients. Default: False.

        Raises:
            ValueError: If a parameter is not found in the circuit during
                bowtie construction.

        Note:
            Initialization performs the following steps:
            1. Constructs bowtie circuits for each parameter
            2. Constructs bowtie circuits for each observable term
            3. Identifies active qubits for each bowtie
            4. Transpiles all circuits for the target simulator
            5. Precomputes non-zero overlap indices and slice objects
        """
        self.qc = qc
        self.VarQITE_gradient = VarQITE_gradient
        self.obs = obs
        self.pbar = pbar
        self.phase_fix = phase_fix
        self.accelerator = accelerator
        self.compute_variance = compute_variance
        self.simulator = AerSimulator(method="statevector", device=accelerator)
        exec = ThreadPoolExecutor(max_workers=10)
        self.simulator.set_options(executor=exec)
        dag = circuit_to_dag(qc)

        parameter_bowties, parameter_active_qubits = zip(
            *[
                remove_idle_qwires(parameter_bowtie(copy(dag), p))
                for p in tqdm(
                    qc.parameters,
                    "Getting Parameter bowties",
                    disable=(self.pbar <= 0),
                )
            ]
        )

        if verbose_init:
            print(
                f"""Parameter bowties are ditributed as:
            {dict(sorted(Counter([len(aq) for aq in parameter_active_qubits]).items()))}
            """
            )

        observable_bowties, observable_active_qubits = zip(
            *[
                remove_idle_qwires(observable_bowtie(qc, term, indices))
                for term, indices, _ in tqdm(
                    obs.to_sparse_list(),
                    "Getting observable bowties",
                    disable=(self.pbar <= 0),
                )
            ]
        )
        if verbose_init:
            print(
                f"""Observable bowties are ditributed as:
            {dict(sorted(Counter([len(aq) for aq in observable_active_qubits]).items()))} """
            )

        self.bowties = parameter_bowties + observable_bowties
        self.active_qubits = parameter_active_qubits + observable_active_qubits

        for bw in tqdm(self.bowties, "Transpiling Circuits", disable=(self.pbar <= 0)):
            bw.save_statevector()
            bw = transpile(bw, self.simulator)

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

    def get_derivatives(self, parameter_dict: dict[Parameter, float], tracking_time: bool = False):
        """Compute the generalized QGT matrix and energy expectation value.

        This is the main computation method that evaluates all bowtie circuits
        at the given parameter values and constructs the generalized QGT matrix.
        The matrix structure is:

        ```
        ┌─────────────┬──────────────┐
        │     QGT     │   Gradient   │
        │  (NxN)      │   (NxM)      │
        ├─────────────┼──────────────┤
        │  Gradient†  │   Variance   │
        │  (MxN)      │   (MxM)      │
        └─────────────┴──────────────┘
        ```

        where N = number of parameters, M = number of observable terms.



        Args:
            parameter_dict: Dictionary mapping circuit Parameters to their
                numerical values. Can be a subset of circuit parameters;
                unspecified parameters remain symbolic.
            tracking_time: If True, returns timing information for profiling.
                Default: False.

        Returns:
            If tracking_time is False:
                tuple: (gen_qgt, energy) where:
                    - gen_qgt: Complex numpy array of shape (N+M, N+M) containing
                      the generalized QGT matrix
                    - energy: Complex number representing the energy expectation
                      value ⟨ψ|H|ψ⟩

            If tracking_time is True:
                tuple: ((gen_qgt, energy), (time_simulation, time_qgt)) where:
                    - time_simulation: Time in seconds for statevector computation
                    - time_qgt: Time in seconds for QGT matrix assembly

        Note:
            The generalized QGT is computed as:
            G_ij = Re[⟨∂_i ψ|∂_j ψ⟩ - ⟨∂_i ψ|ψ⟩⟨ψ|∂_j ψ⟩] / 4

            Gradient entries include appropriate factors of -i for imaginary
            time evolution when VarQITE_gradient=False.
        """
        # Precompute the tensors for QGT and gradient
        bound_circuits = [bw.assign_parameters(parameter_dict, strict=False) for bw in self.bowties]

        timeA = time()
        computed_states = self.simulator.run(bound_circuits).result().results
        statevectors = [result.data.statevector for result in computed_states]
        timeB = time()
        if self.pbar > 0:
            print(f"It took {timeB - timeA} to compute the statevectors")

        # Reshaping the statevectors allows for faster computation of sparse overlaps
        tensors = [
            np.reshape(sv.data, [2] * len(active_qubits))
            for sv, active_qubits in zip(statevectors, self.active_qubits)
        ]
        # Compute QGT
        gen_qgt = np.zeros([len(self.bowties)] * 2, dtype=complex)
        qgt_pbar = tqdm(
            list(zip(self.non_zero_indices, self.slices)), "computing qgt", disable=(self.pbar <= 1)
        )
        qgt_entries = [
            tensor_phase_fix(tensors[i], sliceab, tensors[j], slicebc, self.phase_fix)
            for (i, j), (sliceab, slicebc) in qgt_pbar
        ]

        # We need to add a phase of -1.0j if the entry is part of the gradient and we
        # are computing the imaginary gradient.
        for qgt_entry, indices in zip(qgt_entries, self.non_zero_indices):
            gen_qgt[*indices] = qgt_entry
            if indices[0] >= self.qc.num_parameters:
                gen_qgt[*indices] *= (-1.0j) ** self.VarQITE_gradient
                coeff_index = indices[0] - self.qc.num_parameters
                gen_qgt[*indices] *= self.obs.coeffs[coeff_index]
            if indices[1] >= self.qc.num_parameters:
                gen_qgt[*indices] *= (-1.0j) ** self.VarQITE_gradient
                coeff_index = indices[1] - self.qc.num_parameters
                gen_qgt[*indices] *= self.obs.coeffs[coeff_index]

        # We need to reshape the generalized qgt (we only computed the upper diagonal)
        gen_qgt += np.triu(gen_qgt.conj().T, 1)
        gen_qgt /= 4

        energy = sum(
            [
                zeroth(osv) * coeff
                for osv, coeff in zip(
                    tensors[self.qc.num_parameters :],
                    self.obs.coeffs,
                )
            ]
        )
        timeC = time()
        if tracking_time:
            return ((gen_qgt, energy), (timeB - timeA, timeC - timeB))
        return gen_qgt, energy

    def extract_qgt(self, gen_qgt: np.ndarray):
        """Extract the QGT block from the generalized QGT matrix.

        Extracts the upper-left NxN block containing the Quantum Geometric
        Tensor (metric tensor) for the parameter space.

        Args:
            gen_qgt: Generalized QGT matrix of shape (N+M, N+M) returned by
                [`get_derivatives()`](bowtie_qgt/bowtieqgt.py:113).

        Returns:
            Complex numpy array of shape (N, N) where N is the number of
            circuit parameters. This is the QGT matrix G_ij = ⟨∂_i ψ|∂_j ψ⟩.

        Example:
            >>> gen_qgt, energy = bowtie.get_derivatives(params)
            >>> qgt = bowtie.extract_qgt(gen_qgt)
            >>> print(qgt.shape)  # (num_parameters, num_parameters)
        """
        return gen_qgt[: self.qc.num_parameters, : self.qc.num_parameters]

    def extract_gradient(self, gen_qgt: np.ndarray):
        """Extract the energy gradient from the generalized QGT matrix.

        Extracts and processes the gradient block to compute ∂E/∂θ_i for each
        parameter θ_i. The gradient is computed by summing over observable terms
        with appropriate normalization and sign conventions.

        Args:
            gen_qgt: Generalized QGT matrix of shape (N+M, N+M) returned by
                [`get_derivatives()`](bowtie_qgt/bowtieqgt.py:113).

        Returns:
            Complex numpy array of shape (N,) containing the gradient components
            ∂E/∂θ_i. For VarQITE_gradient=True, returns real gradients; for
            VarQITE_gradient=False, returns imaginary gradients with appropriate
            sign convention.

        Note:
            The factor of 4 accounts for the 1/4 normalization in the QGT
            computation. The sign convention depends on whether real or imaginary
            time evolution is being used.

        Example:
            >>> gen_qgt, energy = bowtie.get_derivatives(params)
            >>> gradient = bowtie.extract_gradient(gen_qgt)
            >>> print(gradient.shape)  # (num_parameters,)
        """
        gradient_sector = gen_qgt[: self.qc.num_parameters, self.qc.num_parameters :]
        return 4 * np.sum(gradient_sector, axis=1) * (-1.0) ** (not self.VarQITE_gradient)

    def extract_variance(self, gen_qgt: np.ndarray, energy: float | None = None):
        """Extract the observable variance from the generalized QGT matrix.

        Extracts the lower-right MxM block and computes the total variance
        Var(H) = ⟨H²⟩ - ⟨H⟩². The variance sector contains ⟨H²⟩ (with phase
        correction when phase_fix=True), so we must subtract ⟨H⟩² (energy²)
        to obtain the actual variance when the energy parameter is provided.

        Args:
            gen_qgt: Generalized QGT matrix of shape (N+M, N+M) returned by
                [`get_derivatives()`](bowtie_qgt/bowtieqgt.py:113).
            energy: The expectation value ⟨H⟩ of the observable. When provided,
                the method returns Var(H) = ⟨H²⟩ - ⟨H⟩². When not provided,
                returns ⟨H²⟩ (useful for backward compatibility).

        Returns:
            Complex number representing the variance of the observable when
            energy is provided, or ⟨H²⟩ when energy is not provided. For
            Hermitian observables, the variance should be real and non-negative.

        Example:
            >>> # Compute variance (recommended)
            >>> gen_qgt, energy = bowtie.get_derivatives(params)
            >>> variance = bowtie.extract_variance(gen_qgt, energy).real
            >>>
            >>> # Backward compatibility: get ⟨H²⟩ without energy
            >>> gen_qgt, _ = bowtie.get_derivatives(params)
            >>> h2_expectation = bowtie.extract_variance(gen_qgt).real
        """
        var_sector = gen_qgt[self.qc.num_parameters :, self.qc.num_parameters :]

        # The variance sector was divided by 4 along with the rest of gen_qgt,
        # but variance should not be divided by 4, so we multiply back
        variance_sum = (-1) ** (self.VarQITE_gradient) * np.sum(var_sector.real) * 4

        # The variance sector contains ⟨H²⟩ (with phase correction applied if phase_fix=True)
        # To get Var(H) = ⟨H²⟩ - ⟨H⟩², we need to subtract energy² when energy is provided
        if energy is not None and not self.phase_fix:
            return variance_sum - energy**2
        if energy is None and not self.phase_fix:
            raise ValueError("Energy must be provided when phase_fix=False")
        else:
            return variance_sum
