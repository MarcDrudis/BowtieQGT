# SPDX-License-Identifier: Apache-2.0
# (C) Copyright IBM 2025.

"""Compare BowtieQGT against Qiskit's ReverseQGT and ReverseEstimatorGradient.

Each case is a different circuit topology and gate set.
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
        # (num_qubits, rotation_blocks, entanglement_blocks, entanglement_pairs)
        # Test 1: Extended linear chain topology with mixed gates
        (
            7,
            ["ry", "rz"],
            ["rxx", "ryy"],
            [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)],
        ),
        (7, ["rx", "ry"], "cx", [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]),
        # Test 2: Star topology with central hub and mixed parametric gates
        (
            8,
            ["rx", "rz"],
            ["rxx", "rzz"],
            [(3, 0), (3, 1), (3, 2), (3, 4), (3, 5), (3, 6), (3, 7)],
        ),
        # Test 3: Ring/circular topology with parametric gates on more qubits
        (
            6,
            ["ry"],
            ["rxx", "ryy", "rzz"],
            [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)],
        ),
        # Test 4: Dense mesh topology (partial all-to-all with mixed gates)
        (
            6,
            ["rx", "ry"],
            ["rxx", "ryy"],
            [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 5), (1, 2), (3, 4)],
        ),
        # Test 5: Multi-layer ladder topology (3 parallel chains with cross-rungs)
        (
            6,
            ["ry", "rz"],
            "cx",
            [(0, 1), (2, 3), (4, 5), (0, 2), (2, 4), (1, 3), (3, 5)],
        ),
        # Test 6: Hexagonal/honeycomb-like topology with parametric gates
        (
            6,
            ["ry", "rz"],
            ["rxx", "rzz"],
            [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3), (1, 4), (2, 5)],
        ),
        # Test 7: Tree topology (binary tree structure)
        (
            7,
            ["ry", "rz"],
            ["ryy", "rzz"],
            [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)],
        ),
        # Test 8: Irregular mesh with long-range connections
        (
            8,
            ["rx", "ry", "rz"],
            "cx",
            [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7), (0, 4), (2, 6), (1, 5), (3, 7)],
        ),
        # Test 9: Bipartite-like topology with cross-connections
        (
            8,
            ["ry"],
            ["rxx", "ryy", "rzz"],
            [(0, 4), (1, 5), (2, 6), (3, 7), (0, 5), (1, 6), (2, 7), (3, 4)],
        ),
        # Test 10: Complex mixed topology with varied entanglement
        (
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
        (5, ["ry"], "cx", [(0, 1), (1, 2), (2, 3), (3, 4)]),
        (5, ["rz"], "cx", [(0, 1), (1, 2), (2, 3), (3, 4)]),
        # Test 12: Non-parameterized gates - Hadamard with parameterized rotations
        (4, ["h", "ry"], ["rxx"], [(0, 1), (1, 2), (2, 3)]),
        (4, ["ry", "h"], "cx", [(0, 1), (2, 3)]),
        # Test 13: Non-parameterized gates - S gate with parameterized rotations
        (4, ["s", "ry"], ["rxx", "ryy"], [(0, 1), (1, 2), (2, 3)]),
        (4, ["ry", "s"], "cx", [(0, 1), (1, 2), (2, 3)]),
        # Test 14: Mixed non-parameterized gates - H, S, and CNOT
        (5, ["h", "ry", "s"], "cx", [(0, 1), (1, 2), (2, 3), (3, 4)]),
        (5, ["s", "rz", "h"], "cx", [(0, 1), (1, 2), (2, 3), (3, 4)]),
        # Test 15: Non-parameterized gates with complex topology
        (6, ["h", "ry"], "cx", [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]),
        (6, ["ry", "s"], "cx", [(0, 2), (1, 3), (2, 4), (3, 5), (0, 5)]),
        # Test 16: All non-parameterized two-qubit gates with single parameterized rotation
        (4, ["ry"], "cx", [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3)]),
        # Test 17: Alternating H and S gates with parameterized rotations
        (5, ["h", "s", "ry"], "cx", [(0, 1), (2, 3)]),
        (5, ["ry", "h", "s"], "cx", [(1, 2), (3, 4)]),
    )
    @unpack
    def test_bowtie_qgt_comprehensive(
        self,
        num_qubits,
        rotation_blocks,
        entanglement_blocks,
        entanglement_pairs,
    ):
        """Compare every block of the generalized QGT against Qiskit references.

        Args:
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
        bowtie = BowtieQGT(qc, obs)

        param_values = np.arange(qc.num_parameters) * 0.1
        generalized_qgt, energy = bowtie.evaluate(param_values)

        reference_qgt = (
            ReverseQGT(derivative_type=DerivativeType.COMPLEX)
            .run([qc], [param_values.tolist()])
            .result()
            .qgts[0]
        )
        reference_state = Statevector(qc.assign_parameters(param_values))
        reference_energy = reference_state.expectation_value(obs).real
        reference_variance = (
            reference_state.expectation_value((obs @ obs).simplify()).real - reference_energy**2
        )
        # Qiskit's REAL/IMAG gradients of <H - E> are -2 b_imag and -2 b_real respectively.
        centered_obs = (obs - reference_energy * SparsePauliOp("I" * num_qubits)).simplify()
        reference_b = {
            which: -0.5
            * ReverseEstimatorGradient(derivative_type)
            .run([qc], [centered_obs], [param_values.tolist()])
            .result()
            .gradients[0]
            for which, derivative_type in [
                ("imag_time", DerivativeType.REAL),
                ("real_time", DerivativeType.IMAG),
            ]
        }

        with self.subTest("Generalized QGT is Hermitian"):
            self.assertTrue(ishermitian(generalized_qgt, atol=1e-12))

        with self.subTest("QGT is positive semi-definite"):
            self.assertTrue(np.all(eigh(bowtie.extract_qgt(generalized_qgt))[0] >= -1e-10))

        with self.subTest("Complex QGT matches reference"):
            np.testing.assert_allclose(
                bowtie.extract_qgt(generalized_qgt, part="complex"),
                reference_qgt,
                rtol=1e-5,
                atol=1e-8,
            )

        with self.subTest("Energy matches reference"):
            self.assertIsInstance(energy, float)
            np.testing.assert_allclose(energy, reference_energy, rtol=1e-5, atol=1e-8)

        for which, expected in reference_b.items():
            with self.subTest(f"{which} gradient matches reference"):
                gradient = bowtie.extract_gradient(generalized_qgt, which=which)
                self.assertEqual(gradient.shape, (qc.num_parameters,))
                np.testing.assert_allclose(gradient, expected, rtol=1e-5, atol=1e-8)

        with self.subTest("Variance matches reference"):
            np.testing.assert_allclose(
                bowtie.extract_variance(generalized_qgt), reference_variance, rtol=1e-5, atol=1e-8
            )


if __name__ == "__main__":
    unittest.main()
