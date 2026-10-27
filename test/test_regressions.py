# SPDX-License-Identifier: Apache-2.0
# (C) Copyright IBM 2025.

"""Regression tests for input validation and edge cases of BowtieQGT."""

import unittest
import warnings

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import QFTGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

from bowtie_qgt.bowtieqgt import BowtieQGT

OBS = SparsePauliOp.from_list([("ZI", 1.0), ("IX", 0.5)])
OBS3 = SparsePauliOp.from_list([("ZZI", 1.0), ("IXX", 0.5)])


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

    def test_classical_bits(self):
        qc = QuantumCircuit(2, 1)
        qc.ry(Parameter("a"), 0)
        with self.assertRaisesRegex(ValueError, "classical bits"):
            BowtieQGT(qc, OBS)

    def test_control_flow(self):
        a = Parameter("a")
        qc = QuantumCircuit(2)
        qc.ry(a, 0)
        with qc.box():
            qc.rx(Parameter("b"), 1)
        with self.assertRaisesRegex(ValueError, "Control-flow operation 'box'"):
            BowtieQGT(qc, OBS)

    def test_parameterized_global_phase(self):
        qc = QuantumCircuit(2)
        qc.ry(Parameter("a"), 0)
        qc.global_phase = Parameter("b")
        with self.assertRaisesRegex(ValueError, "global phase"):
            BowtieQGT(qc, OBS)

    def test_simulator_must_be_noiseless_statevector(self):
        qc = QuantumCircuit(2)
        qc.ry(Parameter("a"), 0)
        noise = NoiseModel()
        noise.add_all_qubit_quantum_error(depolarizing_error(0.05, 1), ["ry"])
        for simulator in [
            AerSimulator(method="density_matrix"),
            AerSimulator(method="statevector", noise_model=noise),
        ]:
            with self.assertRaisesRegex(ValueError, "noiseless AerSimulator"):
                BowtieQGT(qc, OBS, simulator=simulator)

    def test_non_native_gate_is_transpiled(self):
        """A gate Aer cannot run natively must still work (bowties are transpiled)."""
        a = Parameter("a")
        qc = QuantumCircuit(2)
        qc.ry(a, 0)
        qc.append(QFTGate(2), [0, 1])
        _, energy = BowtieQGT(qc, OBS).evaluate({a: 0.3})
        expected = Statevector(qc.assign_parameters({a: 0.3})).expectation_value(OBS)
        self.assertAlmostEqual(energy, expected.real)


def _exact(qc, hamiltonian, values):
    """Energy and variance from the full statevector."""
    state = Statevector(qc.assign_parameters(values))
    energy = state.expectation_value(hamiltonian).real
    variance = state.expectation_value((hamiltonian @ hamiltonian).simplify()).real - energy**2
    return energy, variance


class TestEdgeCases(unittest.TestCase):
    """Cases the reference suite does not exercise."""

    def setUp(self):
        params = ParameterVector("p", 3)
        self.qc = QuantumCircuit(3)
        self.qc.ry(params[0], 0)
        self.qc.ry(params[1], 1)
        self.qc.rx(params[2], 2)
        self.values = [0.4, 0.7, -0.3]

    def test_variance_with_disjoint_light_cones(self):
        """Bowties on disjoint qubits are skipped; the variance must still be exact."""
        hamiltonian = SparsePauliOp.from_list([("IIZ", 1.0), ("IZI", 1.0), ("ZII", 0.5)])
        bowtie = BowtieQGT(self.qc, hamiltonian)
        generalized_qgt, energy = bowtie.evaluate(self.values)
        expected_energy, expected_variance = _exact(self.qc, hamiltonian, self.values)
        self.assertAlmostEqual(energy, expected_energy)
        self.assertAlmostEqual(bowtie.extract_variance(generalized_qgt), expected_variance)

    def test_identity_terms_shift_energy_only(self):
        traceless = SparsePauliOp.from_list([("IIZ", 1.0), ("XXI", 0.5)])
        shifted = SparsePauliOp.from_list([("IIZ", 1.0), ("III", 2.0), ("XXI", 0.5), ("III", -0.5)])
        g_traceless, e_traceless = BowtieQGT(self.qc, traceless).evaluate(self.values)
        bowtie = BowtieQGT(self.qc, shifted)
        g_shifted, e_shifted = bowtie.evaluate(self.values)
        self.assertAlmostEqual(e_shifted, e_traceless + 1.5)
        self.assertAlmostEqual(e_shifted, _exact(self.qc, shifted, self.values)[0])
        np.testing.assert_allclose(g_shifted, g_traceless, atol=1e-12)

    def test_trivial_hamiltonian_warns(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bowtie = BowtieQGT(self.qc, SparsePauliOp("III", 3.0))
        self.assertTrue(any("Trivial Hamiltonian" in str(w.message) for w in caught))
        generalized_qgt, energy = bowtie.evaluate(self.values)
        self.assertAlmostEqual(energy, 3.0)
        np.testing.assert_array_equal(bowtie.extract_gradient(generalized_qgt, "real_time"), 0)
        self.assertEqual(bowtie.extract_variance(generalized_qgt), 0)

    def test_parameter_value_formats(self):
        """Sequences and every key style assign_parameters accepts give the same result."""
        params = ParameterVector("v", 2)
        qc = QuantumCircuit(2)
        qc.ry(params[0], 0)
        qc.rx(params[1], 1)
        bowtie = BowtieQGT(qc, OBS)
        expected = bowtie.evaluate([0.1, 0.2])[1]
        for values in (
            {params: [0.1, 0.2]},
            {"v[0]": 0.1, "v[1]": 0.2},
            dict(zip(params, [0.1, 0.2])),
        ):
            self.assertAlmostEqual(bowtie.evaluate(values)[1], expected)
        with self.assertRaisesRegex(ValueError, "Expected 2 parameter values"):
            bowtie.evaluate([0.1])

    def test_no_bowties(self):
        """No parameters and a trivial Hamiltonian: nothing to simulate."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bowtie = BowtieQGT(QuantumCircuit(2), SparsePauliOp("II", 1.5))
        generalized_qgt, energy = bowtie.evaluate([])
        self.assertEqual(generalized_qgt.shape, (0, 0))
        self.assertAlmostEqual(energy, 1.5)

    def test_missing_parameter_value(self):
        bowtie = BowtieQGT(self.qc, OBS3)
        with self.assertRaisesRegex(ValueError, "Missing values"):
            bowtie.evaluate({self.qc.parameters[0]: 0.1})

    def test_invalid_choices(self):
        bowtie = BowtieQGT(self.qc, OBS3)
        generalized_qgt, _ = bowtie.evaluate(self.values)
        with self.assertRaisesRegex(ValueError, "which"):
            bowtie.extract_gradient(generalized_qgt, which="real")
        with self.assertRaisesRegex(ValueError, "part"):
            bowtie.extract_qgt(generalized_qgt, part="abs")

    def test_timings_recorded(self):
        bowtie = BowtieQGT(self.qc, OBS3)
        bowtie.evaluate(self.values)
        self.assertEqual(set(bowtie.last_timings), {"simulation", "assembly"})


if __name__ == "__main__":
    unittest.main()
