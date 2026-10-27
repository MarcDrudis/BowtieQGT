# SPDX-License-Identifier: Apache-2.0
# (C) Copyright IBM 2025.

"""Tests that BowtieQGT rejects circuits it cannot handle at construction time."""

import unittest

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import QFTGate
from qiskit.quantum_info import SparsePauliOp, Statevector

from bowtie_qgt.bowtieqgt import BowtieQGT

OBS = SparsePauliOp.from_list([("ZI", 1.0), ("IX", 0.5)])


class TestValidation(unittest.TestCase):
    """Construction-time validation of the circuit."""

    def test_parameter_in_two_gates(self):
        a = Parameter("a")
        qc = QuantumCircuit(2)
        qc.ry(a, 0)
        qc.rx(a, 1)
        with self.assertRaisesRegex(ValueError, "more than one gate"):
            BowtieQGT(qc, OBS)

    def test_parameter_expression(self):
        a = Parameter("a")
        qc = QuantumCircuit(2)
        qc.ry(2 * a, 0)
        with self.assertRaisesRegex(ValueError, "expressions are not supported"):
            BowtieQGT(qc, OBS)

    def test_unsupported_parameterized_gate(self):
        a = Parameter("a")
        qc = QuantumCircuit(2)
        qc.p(a, 0)
        with self.assertRaisesRegex(ValueError, "'p' is not supported.*rx, ry, rz"):
            BowtieQGT(qc, OBS)

    def test_non_native_gate_is_transpiled(self):
        """A gate Aer cannot run natively must still work (bowties are transpiled)."""
        a = Parameter("a")
        qc = QuantumCircuit(2)
        qc.ry(a, 0)
        qc.append(QFTGate(2), [0, 1])
        _, energy = BowtieQGT(qc, OBS).get_derivatives({a: 0.3})
        expected = Statevector(qc.assign_parameters({a: 0.3})).expectation_value(OBS)
        self.assertAlmostEqual(complex(energy), complex(expected))


if __name__ == "__main__":
    unittest.main()
