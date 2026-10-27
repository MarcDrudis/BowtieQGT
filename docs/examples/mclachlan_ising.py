# SPDX-License-Identifier: Apache-2.0
# (C) Copyright IBM 2025.

"""VarQRTE and VarQITE of a transverse-field Ising chain with McLachlan's principle.

H = -J Σ Z_i Z_{i+1} - h Σ X_i on an open chain. For h < J the ground state is
ferromagnetically ordered along Z; for h > J it is a paramagnet polarized along X; the
quantum phase transition sits at h = J. We take h = J.

Real time: quench from |0...0⟩ (all spins up along Z) and follow the magnetization
m_z(t) = ⟨Σ Z_i⟩ / n. The transverse field rotates the spins away from Z and m_z relaxes
to zero. The quench also builds up entanglement, which a fixed-depth ansatz can only hold
for so long: with 3 layers VarQRTE tracks the exact curve until t ≈ 1.3 and then drifts.
More layers push that point later.

Imaginary time: start from |+...+⟩ (the h → ∞ ground state) and flow to the ground state.
The energy decreases monotonically and Var[H] goes to zero, which certifies that we reached
an eigenstate rather than just a low energy.

Both evolutions solve McLachlan's equation Re(G) θ̇ = b at every step. Runs in a few
minutes on a laptop.
"""

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import eigsh, expm_multiply

from bowtie_qgt.bowtieqgt import BowtieQGT

n, J, h = 8, 1.0, 1.0
hamiltonian = SparsePauliOp.from_sparse_list(
    [("ZZ", [i, i + 1], -J) for i in range(n - 1)] + [("X", [i], -h) for i in range(n)],
    num_qubits=n,
)
magnetization = SparsePauliOp.from_sparse_list([("Z", [i], 1 / n) for i in range(n)], n)


def brickwork_ansatz(layers: int, start_in_plus: bool) -> QuantumCircuit:
    """Trotter-like ansatz: alternating rzz bricks and rx/ry layers.

    With all angles zero it prepares |0...0⟩, or |+...+⟩ if start_in_plus.
    """
    qc = QuantumCircuit(n)
    if start_in_plus:
        qc.h(range(n))
    theta = iter(ParameterVector("θ", layers * (n - 1 + 2 * n)))
    for _ in range(layers):
        for offset in (0, 1):
            for i in range(offset, n - 1, 2):
                qc.rzz(next(theta), i, i + 1)
        for i in range(n):
            qc.rx(next(theta), i)
            qc.ry(next(theta), i)
    return qc


def mclachlan_rhs(bowtie: BowtieQGT, which: str, regularization: float = 1e-6):
    """θ̇ = (Re G + λ I)^{-1} b. The small λ handles the redundant directions of G."""

    def rhs(_t, theta):
        generalized_qgt, _ = bowtie.evaluate(theta)
        qgt = bowtie.extract_qgt(generalized_qgt)  # real part, as McLachlan requires
        b = bowtie.extract_gradient(generalized_qgt, which=which)
        return np.linalg.solve(qgt + regularization * np.eye(len(b)), b)

    return rhs


# --- VarQRTE: magnetization after a quench --------------------------------------------
ansatz = brickwork_ansatz(layers=3, start_in_plus=False)
bowtie = BowtieQGT(ansatz, hamiltonian)
times = np.linspace(0, 3, 41)
solution = solve_ivp(
    mclachlan_rhs(bowtie, "real_time"),
    (times[0], times[-1]),
    np.zeros(ansatz.num_parameters),
    t_eval=times,
    rtol=1e-6,
    atol=1e-8,
)
m_varqrte = [
    Statevector(ansatz.assign_parameters(theta)).expectation_value(magnetization).real
    for theta in solution.y.T
]
# Exact reference: |ψ(t)⟩ = exp(-iHt)|0...0⟩ on the full 2^n state vector.
H_sparse = hamiltonian.to_matrix(sparse=True)
psi0 = np.zeros(2**n, dtype=complex)
psi0[0] = 1
exact_states = expm_multiply(-1j * H_sparse, psi0, start=times[0], stop=times[-1], num=len(times))
m_exact = [Statevector(psi).expectation_value(magnetization).real for psi in exact_states]

# --- VarQITE: ground-state preparation ------------------------------------------------
ansatz = brickwork_ansatz(layers=4, start_in_plus=True)
bowtie = BowtieQGT(ansatz, hamiltonian)
taus = np.linspace(0, 5, 41)
solution = solve_ivp(
    mclachlan_rhs(bowtie, "imag_time"),
    (taus[0], taus[-1]),
    np.zeros(ansatz.num_parameters),
    t_eval=taus,
    rtol=1e-6,
    atol=1e-8,
)
energies, variances = [], []
for theta in solution.y.T:
    generalized_qgt, energy = bowtie.evaluate(theta)
    energies.append(energy)
    variances.append(bowtie.extract_variance(generalized_qgt))
ground_energy = eigsh(H_sparse, k=1, which="SA")[0][0]
print(f"VarQITE energy {energies[-1]:.5f}, exact ground energy {ground_energy:.5f}")
print(f"Max |m_varqrte - m_exact| = {np.max(np.abs(np.subtract(m_varqrte, m_exact))):.2e}")

# --- Plots ----------------------------------------------------------------------------
fig, (ax_rt, ax_e, ax_var) = plt.subplots(1, 3, figsize=(13, 3.6), layout="constrained")
ax_rt.plot(times, m_exact, "k-", label="exact")
ax_rt.plot(times, m_varqrte, "o", ms=4, label="VarQRTE")
ax_rt.set(xlabel="t", ylabel=r"$m_z$", title="Quench from $|0\\ldots0\\rangle$")
ax_rt.legend()
ax_e.plot(taus, energies, "o-", ms=3, label="VarQITE")
ax_e.axhline(ground_energy, color="k", ls="--", label="ground energy")
ax_e.set(xlabel=r"$\tau$", ylabel="E", title="Energy")
ax_e.legend()
ax_var.semilogy(taus, variances, "o-", ms=3)
ax_var.set(xlabel=r"$\tau$", ylabel="Var[H]", title="Energy variance")
fig.savefig("mclachlan_ising.png", dpi=150)
