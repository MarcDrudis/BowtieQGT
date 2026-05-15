# SPDX-License-Identifier: Apache-2.0
# (C) Copyright IBM 2025.

"""Utility functions for Bowtie Quantum Geometric Tensor computations.

This module provides helper functions for computing parameter and observable
bowtie circuits, which are essential for quantum geometric tensor calculations.
"""

from qiskit import QuantumCircuit
from qiskit.circuit import Gate, Parameter
from qiskit.circuit.commutation_library import SessionCommutationChecker as scc
from qiskit.circuit.library import PauliGate, XGate, YGate, ZGate
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes.optimization.light_cone import LightCone

# Mapping of rotation gate names to their corresponding Pauli gate objects
_rotation_gates: dict[str, Gate] = {
    "rx": XGate(),
    "ry": YGate(),
    "rz": ZGate(),
    "rzz": PauliGate("ZZ"),
    "rxx": PauliGate("XX"),
    "ryy": PauliGate("YY"),
}


def _extend_lightcone(
    dag: DAGCircuit, light_cone: set[int], light_cone_ops: list[tuple[Gate, tuple]]
) -> DAGCircuit:
    """Extend the light cone backwards through the circuit.

    This method takes an initial light cone at the end of the DAGCircuit
    and extends it backwards by identifying operations that do not commute
    with the light cone operations. Operations that commute are removed.

    Args:
        dag: The directed acyclic graph representation of the quantum circuit.
        light_cone: Set of qubit indices currently in the light cone.
        light_cone_ops: List of tuples containing (operation, qubits) that
            define the light cone operations.

    Returns:
        The modified DAGCircuit with operations outside the light cone removed.

    Note:
        This function modifies the input DAG in place and also returns it.
    """
    # iterate over the nodes in reverse topological order
    for node in reversed(list(dag.topological_op_nodes())):
        if not light_cone.intersection(node.qargs):
            dag.remove_op_node(node)
            continue

        commutes_bool = True
        for op in light_cone_ops:
            commute_bool = scc.commute(op[0], op[1], [], node.op, node.qargs, [])
            if not commute_bool:
                light_cone.update(node.qargs)
                light_cone_ops.append((node.op, node.qargs))
                commutes_bool = False
                break

        if commutes_bool:
            dag.remove_op_node(node)
    return dag


def parameter_bowtie(dag: DAGCircuit, target_parameter: Parameter) -> QuantumCircuit:
    """Compute the auxiliary bowtie circuit for a given parameter.

    The bowtie circuit is constructed by:
    1. Finding the gate containing the target parameter
    2. Computing the light cone of that gate
    3. Sandwiching the generator of the rotation gate between the light cone
       circuit and its inverse

    Args:
        dag: The directed acyclic graph representation of the quantum circuit.
        target_parameter: The parameter for which to compute the bowtie circuit.

    Returns:
        The full parameter bowtie circuit, which may contain some idle quantum
        registers that can be removed using `remove_idle_qwires`.

    Raises:
        ValueError: If the target parameter is not found in any gate in the circuit.

    Example:
        >>> from qiskit import QuantumCircuit
        >>> from qiskit.circuit import Parameter
        >>> from qiskit.converters import circuit_to_dag
        >>> theta = Parameter('θ')
        >>> qc = QuantumCircuit(2)
        >>> qc.rx(theta, 0)
        >>> dag = circuit_to_dag(qc)
        >>> bowtie = parameter_bowtie(dag, theta)
    """
    for node in reversed(list(dag.topological_op_nodes())):
        if getattr(node.op, "params", None) == [target_parameter]:
            light_cone = {q for q in node.qargs}
            light_cone_ops = [(node.op, node.qargs)]
            qc = dag_to_circuit(_extend_lightcone(dag, light_cone, light_cone_ops))
            full_circuit = qc.copy()
            derivative = _rotation_gates[node.name]
            full_circuit.compose(derivative, node.qargs, inplace=True)
            full_circuit.compose(qc.inverse(), inplace=True)
            return full_circuit

        else:
            dag.remove_op_node(node)

    raise ValueError(f"Did not find parameter {target_parameter}")


def observable_bowtie(qc: QuantumCircuit, bit_terms: str, indices: list[int]) -> QuantumCircuit:
    """Compute the auxiliary bowtie circuit for a given observable.

    The bowtie circuit is constructed by:
    1. Computing the light cone of the observable
    2. Sandwiching the Pauli gate corresponding to the observable between
       the light cone circuit and its inverse

    Args:
        qc: The quantum circuit for which to compute the observable bowtie.
        bit_terms: Pauli string representing the observable terms (e.g., "XYZ"
            for X⊗Y⊗Z). The string is reversed internally to match Qiskit's
            qubit ordering convention.
        indices: List of qubit indices on which the Pauli terms act, ordered
            to correspond with the bit_terms string.

    Returns:
        The full observable bowtie circuit, which may contain some idle quantum
        registers that can be removed using `remove_idle_qwires`.

    Example:
        >>> from qiskit import QuantumCircuit
        >>> qc = QuantumCircuit(3)
        >>> qc.h(0)
        >>> qc.cx(0, 1)
        >>> bowtie = observable_bowtie(qc, "ZZ", [0, 1])

    Note:
        TODO: Use DAG representation instead of circuit representation for
        consistency with parameter_bowtie.
    """
    light_cone_circuit = PassManager(LightCone(bit_terms, indices)).run([qc])[0]
    bowtie_circuit = light_cone_circuit.copy()
    bowtie_circuit.compose(PauliGate(bit_terms[::-1]), qubits=indices, inplace=True)
    bowtie_circuit.compose(light_cone_circuit.inverse(), inplace=True)
    return bowtie_circuit


def remove_idle_qwires(qc: QuantumCircuit) -> tuple[QuantumCircuit, tuple[int, ...]]:
    """Remove idle quantum wires from a circuit.

    This function identifies and removes quantum wires (qubits) that have no
    operations applied to them throughout the circuit, returning a reduced
    circuit and the indices of the active wires.

    Args:
        qc: The quantum circuit from which to remove idle wires.

    Returns:
        A tuple containing:
        - The reduced quantum circuit with idle wires removed
        - A tuple of indices indicating which qubits from the original circuit
          remain in the reduced circuit (in order)

    Example:
        >>> from qiskit import QuantumCircuit
        >>> qc = QuantumCircuit(5)
        >>> qc.h(0)
        >>> qc.cx(0, 2)
        >>> reduced_qc, active_indices = remove_idle_qwires(qc)
        >>> print(f"Active qubits: {active_indices}")
        Active qubits: (0, 2)
        >>> print(f"Reduced circuit has {reduced_qc.num_qubits} qubits")
        Reduced circuit has 2 qubits
    """
    dag = circuit_to_dag(qc)
    idle_wires = list(dag.idle_wires())
    idle_wires_indices = [qc.find_bit(q).index for q in idle_wires]
    active_wires_indices = tuple(w for w in range(qc.num_qubits) if w not in idle_wires_indices)
    dag.remove_qubits(*idle_wires)
    circuit = dag_to_circuit(dag)
    return circuit, active_wires_indices
