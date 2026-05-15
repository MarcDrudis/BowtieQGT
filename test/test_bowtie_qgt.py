# SPDX-License-Identifier: Apache-2.0
# (C) Copyright IBM 2025.

"""Unit tests for BowtieQGT without external dependencies.

This test module validates the BowtieQGT implementation by comparing its results
against Qiskit's reference implementations (ReverseQGT and ReverseEstimatorGradient).
It uses various circuit topologies and gate types.
"""

import unittest

import numpy as np
from ddt import data, ddt, unpack
from numpy.linalg import eigh
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_algorithms.gradients import DerivativeType, ReverseEstimatorGradient, ReverseQGT
from scipy.linalg import ishermitian

from bowtie_qgt.bowtieqgt import BowtieQGT


def create_test_circuit(
    num_qubits: int,
    num_layers: int,
    rotation_blocks: list[str] | str | None = None,
    entanglement_blocks: list[str] | str | None = None,
    entanglement_pairs: list[tuple[int, int]] | None = None,
) -> QuantumCircuit:
    """Create a parameterized quantum circuit with configurable gates.

    Constructs a circuit with alternating single-qubit rotations and
    two-qubit entangling gates using the specified gate types and
    entanglement pairs.

    Args:
        num_qubits: Number of qubits in the circuit.
        num_layers: Number of layers of gates to apply.
        rotation_blocks: Single-qubit rotation gates to use. Can be a list
            like ['ry', 'rz'] or a single gate like 'ry'. If None, defaults
            to ['ry', 'rz'].
        entanglement_blocks: Two-qubit entangling gates to use. Can be a list
            like ['rxx', 'ryy', 'rzz'] or a single gate like 'cx'. If None,
            defaults to ['rxx', 'ryy', 'rzz'].
        entanglement_pairs: List of qubit index pairs for entanglement.
            If None, defaults to brick-layer pattern.

    Returns:
        Parameterized QuantumCircuit with specified gates and entanglement.
    """
    # Set defaults if not provided
    if rotation_blocks is None:
        rotation_blocks = ["ry", "rz"]
    if entanglement_blocks is None:
        entanglement_blocks = ["rxx", "ryy", "rzz"]

    # Normalize to lists
    if isinstance(rotation_blocks, str):
        rotation_blocks = [rotation_blocks]
    if isinstance(entanglement_blocks, str):
        entanglement_blocks = [entanglement_blocks]

    qc = QuantumCircuit(num_qubits)

    # Count parameters needed
    num_rotation_gates = len(rotation_blocks)
    num_entangling_gates = len(entanglement_blocks)

    # Use provided pairs or default brick-layer pattern
    if entanglement_pairs is None:
        # Default: brick-layer pattern changes per layer
        total_params = 0
        for layer in range(num_layers):
            total_params += num_rotation_gates * num_qubits
            offset = layer % 2
            num_pairs = len([(q, q + 1) for q in range(offset, num_qubits - 1, 2)])
            total_params += num_entangling_gates * num_pairs
        total_params += num_rotation_gates * num_qubits  # Final layer
    else:
        # Fixed pairs for all layers
        num_pairs = len(entanglement_pairs)
        total_params = num_layers * (
            num_rotation_gates * num_qubits + num_entangling_gates * num_pairs
        )
        total_params += num_rotation_gates * num_qubits  # Final layer

    params = ParameterVector("θ", total_params)
    param_idx = 0

    # Apply layers of rotations and entangling gates
    for layer in range(num_layers):
        # Single-qubit rotations
        for q in range(num_qubits):
            for gate_name in rotation_blocks:
                if gate_name == "h":
                    qc.h(q)
                elif gate_name == "s":
                    qc.s(q)
                else:
                    if gate_name == "rx":
                        qc.rx(params[param_idx], q)
                    elif gate_name == "ry":
                        qc.ry(params[param_idx], q)
                    elif gate_name == "rz":
                        qc.rz(params[param_idx], q)
                    param_idx += 1

        # Two-qubit entangling gates
        if entanglement_pairs is None:
            # Brick-layer pattern
            offset = layer % 2
            pairs = [(q, q + 1) for q in range(offset, num_qubits - 1, 2)]
        else:
            pairs = entanglement_pairs

        for q1, q2 in pairs:
            for gate_name in entanglement_blocks:
                if gate_name == "cx":
                    qc.cx(q1, q2)
                else:
                    if gate_name == "rxx":
                        qc.rxx(params[param_idx], q1, q2)
                    elif gate_name == "ryy":
                        qc.ryy(params[param_idx], q1, q2)
                    elif gate_name == "rzz":
                        qc.rzz(params[param_idx], q1, q2)
                    param_idx += 1

    # Final layer of single-qubit rotations
    for q in range(num_qubits):
        for gate_name in rotation_blocks:
            if gate_name == "h":
                qc.h(q)
            elif gate_name == "s":
                qc.s(q)
            else:
                if gate_name == "rx":
                    qc.rx(params[param_idx], q)
                elif gate_name == "ry":
                    qc.ry(params[param_idx], q)
                elif gate_name == "rz":
                    qc.rz(params[param_idx], q)
                param_idx += 1

    return qc


def create_test_observable(num_qubits: int) -> SparsePauliOp:
    """Create a test Hamiltonian observable.

    Constructs a Hamiltonian with nearest-neighbor XX, YY, ZZ interactions
    and single-qubit Z and X terms.

    Args:
        num_qubits: Number of qubits for the observable.

    Returns:
        SparsePauliOp representing the test Hamiltonian.
    """
    pauli_list = []

    # Nearest-neighbor two-qubit interactions
    for q in range(num_qubits - 1):
        for pauli in ["XX", "YY", "ZZ"]:
            pauli_str = "I" * q + pauli + "I" * (num_qubits - q - 2)
            pauli_list.append((pauli_str, -1.0))

    # Single-qubit Z terms
    for q in range(num_qubits):
        pauli_str = "I" * q + "Z" + "I" * (num_qubits - q - 1)
        pauli_list.append((pauli_str, -1.0))

    # Single-qubit X terms
    for q in range(num_qubits):
        pauli_str = "I" * q + "X" + "I" * (num_qubits - q - 1)
        pauli_list.append((pauli_str, 0.25))

    return SparsePauliOp.from_list(pauli_list)


@ddt
class TestBowtieQGT(unittest.TestCase):
    """Test suite for BowtieQGT implementation."""

    @data(
        # (VarQITE, phase_fix, num_qubits, rotation_blocks, entanglement_blocks, entanglement_pairs)
        # Test 1: Extended linear chain topology with mixed gates
        (
            True,
            True,
            7,
            ["ry", "rz"],
            ["rxx", "ryy"],
            [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)],
        ),
        (False, True, 7, ["rx", "ry"], "cx", [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]),
        # Test 2: Star topology with central hub and mixed parametric gates
        (
            True,
            True,
            8,
            ["rx", "rz"],
            ["rxx", "rzz"],
            [(3, 0), (3, 1), (3, 2), (3, 4), (3, 5), (3, 6), (3, 7)],
        ),
        # Test 3: Ring/circular topology with parametric gates on more qubits
        (
            True,
            True,
            6,
            ["ry"],
            ["rxx", "ryy", "rzz"],
            [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)],
        ),
        # Test 4: Dense mesh topology (partial all-to-all with mixed gates)
        (
            False,
            True,
            6,
            ["rx", "ry"],
            ["rxx", "ryy"],
            [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 5), (1, 2), (3, 4)],
        ),
        # Test 5: Multi-layer ladder topology (3 parallel chains with cross-rungs)
        (
            True,
            True,
            6,
            ["ry", "rz"],
            "cx",
            [(0, 1), (2, 3), (4, 5), (0, 2), (2, 4), (1, 3), (3, 5)],
        ),
        # Test 6: Hexagonal/honeycomb-like topology with parametric gates
        (
            True,
            False,
            6,
            ["ry", "rz"],
            ["rxx", "rzz"],
            [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3), (1, 4), (2, 5)],
        ),
        # Test 7: Tree topology (binary tree structure)
        (
            False,
            False,
            7,
            ["ry", "rz"],
            ["ryy", "rzz"],
            [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)],
        ),
        # Test 8: Irregular mesh with long-range connections
        (
            True,
            True,
            8,
            ["rx", "ry", "rz"],
            "cx",
            [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7), (0, 4), (2, 6), (1, 5), (3, 7)],
        ),
        # Test 9: Bipartite-like topology with cross-connections
        (
            False,
            True,
            8,
            ["ry"],
            ["rxx", "ryy", "rzz"],
            [(0, 4), (1, 5), (2, 6), (3, 7), (0, 5), (1, 6), (2, 7), (3, 4)],
        ),
        # Test 10: Complex mixed topology with varied entanglement
        (
            True,
            False,
            7,
            ["rx", "ry"],
            ["rxx", "ryy"],
            [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (0, 3),
                (1, 4),
                (2, 5),
                (3, 6),
                (0, 6),
            ],
        ),
        # Test 11: Non-parameterized gates - CNOT with parameterized rotations
        (True, True, 5, ["ry"], "cx", [(0, 1), (1, 2), (2, 3), (3, 4)]),
        (False, True, 5, ["rz"], "cx", [(0, 1), (1, 2), (2, 3), (3, 4)]),
        # Test 12: Non-parameterized gates - Hadamard with parameterized rotations
        (True, True, 4, ["h", "ry"], ["rxx"], [(0, 1), (1, 2), (2, 3)]),
        (False, True, 4, ["ry", "h"], "cx", [(0, 1), (2, 3)]),
        # Test 13: Non-parameterized gates - S gate with parameterized rotations
        (True, True, 4, ["s", "ry"], ["rxx", "ryy"], [(0, 1), (1, 2), (2, 3)]),
        (False, True, 4, ["ry", "s"], "cx", [(0, 1), (1, 2), (2, 3)]),
        # Test 14: Mixed non-parameterized gates - H, S, and CNOT
        (True, True, 5, ["h", "ry", "s"], "cx", [(0, 1), (1, 2), (2, 3), (3, 4)]),
        (False, True, 5, ["s", "rz", "h"], "cx", [(0, 1), (1, 2), (2, 3), (3, 4)]),
        # Test 15: Non-parameterized gates with complex topology
        (True, True, 6, ["h", "ry"], "cx", [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]),
        (False, False, 6, ["ry", "s"], "cx", [(0, 2), (1, 3), (2, 4), (3, 5), (0, 5)]),
        # Test 16: All non-parameterized two-qubit gates with single parameterized rotation
        (True, True, 4, ["ry"], "cx", [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3)]),
        # Test 17: Alternating H and S gates with parameterized rotations
        (True, True, 5, ["h", "s", "ry"], "cx", [(0, 1), (2, 3)]),
        (False, True, 5, ["ry", "h", "s"], "cx", [(1, 2), (3, 4)]),
    )
    @unpack
    def test_bowtie_qgt_comprehensive(
        self,
        is_VarQITE,
        phase_fix,
        num_qubits,
        rotation_blocks,
        entanglement_blocks,
        entanglement_pairs,
    ):
        """Comprehensive test of BowtieQGT functionality.

        Tests QGT computation, gradient extraction, variance extraction, and
        comparison against reference implementations in a single test to avoid
        redundant computations. Tests various circuit topologies and gate types.

        Args:
            is_VarQITE: If True, uses real gradients; if False, uses imaginary gradients.
            phase_fix: If True, applies phase correction for numerical stability.
            num_qubits: Number of qubits in the test circuit.
            rotation_blocks: Single-qubit rotation gates to use.
            entanglement_blocks: Two-qubit entangling gates to use.
            entanglement_pairs: List of qubit index pairs for entanglement.
        """
        qc = create_test_circuit(
            num_qubits=num_qubits,
            num_layers=2,
            rotation_blocks=rotation_blocks,
            entanglement_blocks=entanglement_blocks,
            entanglement_pairs=entanglement_pairs,
        )
        obs = create_test_observable(num_qubits)

        bowtie = BowtieQGT(
            qc, obs, pbar=0, accelerator="CPU", VarQITE_gradient=is_VarQITE, phase_fix=phase_fix
        )

        # Use sequential parameter values
        param_values = np.arange(qc.num_parameters) * 0.1
        param_dict = dict(zip(qc.parameters, param_values))

        # Compute with BowtieQGT (single computation for all tests)
        gen_qgt, energy = bowtie.get_derivatives(param_dict)
        qgt = bowtie.extract_qgt(gen_qgt)  # type: ignore
        gradient = bowtie.extract_gradient(gen_qgt)  # type: ignore
        variance = bowtie.extract_variance(gen_qgt, energy)  # type: ignore

        # Compute reference values
        reference_qgt = (
            ReverseQGT(derivative_type=DerivativeType.COMPLEX)
            .run([qc], [param_values.tolist()])
            .result()
            .qgts[0]
        )
        reference_state = Statevector(qc.assign_parameters(param_dict))
        reference_energy = reference_state.expectation_value(obs)
        reference_h2 = reference_state.expectation_value((obs**2).simplify())
        reference_variance = reference_h2 - reference_energy**2

        # Compute reference gradient
        derivative_type = DerivativeType.REAL if is_VarQITE else DerivativeType.IMAG
        identity_string = "I" * obs.num_qubits  # type: ignore
        mod_obs = SparsePauliOp.from_list(
            obs.to_list() + [(identity_string, -energy)]  # type: ignore
        ).simplify()
        reference_gradient = (
            ReverseEstimatorGradient(derivative_type)
            .run([qc], [mod_obs], [param_values.tolist()])
            .result()
            .gradients[0]
        )

        # Test 1: QGT is Hermitian
        with self.subTest("Generalized QGT is Hermitian"):
            self.assertTrue(
                ishermitian(gen_qgt.real),  # type: ignore
                (
                    "Real part of generalized QGT should be Hermitian "
                    f"(VarQITE={is_VarQITE}, phase_fix={phase_fix})"
                ),
            )

        # Test 2: QGT is positive semi-definite (only when phase_fix=True)
        if phase_fix:
            with self.subTest("QGT is positive semi-definite"):
                eigenvalues, _ = eigh(qgt.real)
                self.assertTrue(
                    np.all(eigenvalues >= -1e-10),
                    "QGT eigenvalues should be non-negative "
                    f"(VarQITE={is_VarQITE}, phase_fix={phase_fix})",
                )

        # Test 3: QGT real part matches reference (only when phase_fix=True)
        if phase_fix:
            with self.subTest("QGT real part matches reference"):
                np.testing.assert_allclose(
                    qgt.real,
                    reference_qgt.real,
                    rtol=1e-5,
                    atol=1e-8,
                    err_msg=f"QGT real part mismatch (VarQITE={is_VarQITE}, phase_fix={phase_fix})",
                )

        # Test 4: QGT imaginary part matches reference (only when phase_fix=True)
        if phase_fix:
            with self.subTest("QGT imaginary part matches reference"):
                np.testing.assert_allclose(
                    qgt.imag,
                    reference_qgt.imag,
                    rtol=1e-5,
                    atol=1e-8,
                    err_msg=(
                        "QGT imaginary part mismatch"
                        f" (VarQITE={is_VarQITE}, phase_fix={phase_fix})",
                    ),
                )

        # Test 5: Energy matches reference
        with self.subTest("Energy expectation value matches reference"):
            np.testing.assert_allclose(
                energy,
                reference_energy,
                rtol=1e-5,
                atol=1e-8,
                err_msg=f"Energy mismatch (VarQITE={is_VarQITE}, phase_fix={phase_fix})",
            )

        # Test 6: Gradient has correct shape
        with self.subTest("Gradient has correct shape"):
            self.assertEqual(
                gradient.shape, (qc.num_parameters,), "Gradient should have shape (num_parameters,)"
            )

        # Test 7: Gradient is finite
        with self.subTest("Gradient contains finite values"):
            self.assertTrue(
                np.all(np.isfinite(gradient)), "Gradient should contain only finite values"
            )

        # Test 8: Variance is scalar
        with self.subTest("Variance is scalar"):
            self.assertTrue(
                np.isscalar(variance) or variance.shape == (), "Variance should be a scalar value"
            )

        # Test 9: Variance is real
        with self.subTest("Variance is real"):
            self.assertTrue(
                np.abs(np.imag(variance)) < 1e-10,  # type: ignore
                "Variance should be real for Hermitian observables",
            )

        # Test 10: Variance is non-negative
        with self.subTest("Variance is non-negative"):
            self.assertGreaterEqual(
                float(np.real(variance)),  # type: ignore
                -1e-10,
                "Variance should be non-negative",
            )

        # Test 11: Variance matches reference value
        with self.subTest("Variance matches reference value"):
            np.testing.assert_allclose(
                variance,
                reference_variance,
                rtol=1e-5,
                atol=1e-8,
                err_msg="Variance does not match reference value",
            )

        # Test 13: Gradient matches reference (only when phase_fix=True)
        if phase_fix:
            with self.subTest("Gradient matches reference"):
                np.testing.assert_allclose(
                    gradient.real,
                    reference_gradient.real,
                    rtol=1e-5,
                    atol=1e-8,
                    err_msg=(
                        "Gradient real part mismatch "
                        f"(VarQITE={is_VarQITE}, phase_fix={phase_fix})",
                    ),
                )


if __name__ == "__main__":
    unittest.main()
