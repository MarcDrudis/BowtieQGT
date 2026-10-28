# SPDX-License-Identifier: Apache-2.0
# (C) Copyright IBM 2025.

"""Real- and imaginary-time evolution of a transverse-field Ising chain.

Two separate runs, both with McLachlan's variational principle:

- Real time (VarQRTE): |ψ(t)⟩ = e^{-iHt}|ψ0⟩. Unitary dynamics, as in an experiment.
  Solved with extract_gradient(..., which="real_time").
- Imaginary time (VarQITE): |ψ(τ)⟩ ∝ e^{-Hτ}|ψ0⟩. Not physical dynamics but a cooling
  procedure: excited states decay as e^{-(E_k - E_0)τ}, leaving the ground state.
  Solved with extract_gradient(..., which="imag_time").

H = -J Σ Z_i Z_{i+1} - h Σ X_i on an open chain of n spins. For h < J the ground state is
ferromagnetically ordered along Z, for h > J it is a paramagnet polarized along X, and
h = J is the quantum critical point.

1. Real time (J = 0.25, h = 1): quench from |0...0⟩ (all spins along +Z). The strong field
makes every spin precess about X, so the magnetization m_z(t) = ⟨Σ Z_i⟩ / n swings from
+1 down to about -0.9 and back, with period ~π/h. The weak ZZ coupling spreads correlations along
the chain and dephases the precession, so the oscillations shrink and lose regularity.

2. Imaginary time (J = h = 1): start from |+...+⟩, the h → ∞ ground state, and flow to the
critical ground state. The energy decreases monotonically and Var[H] goes to zero, which
certifies that we reached an eigenstate rather than just a low energy.

Both runs solve McLachlan's equation Re(G) θ̇ = b at every step; only b changes.

The ansatz alternates rx/ry rotations with rzz bricks: a Trotter circuit with free angles.
It contains e^{-iHt} at short times and keeps every light cone narrow, so it suits the
bowtie method. Hardware-efficient ansätze with fixed CNOT chains (e.g. efficient_su2)
spread each light cone over the whole chain and track this quench much less accurately.

Runs in about seven minutes on a laptop.
"""

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import n_local
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import eigsh, expm_multiply

from bowtie_qgt.bowtieqgt import BowtieQGT

n = 8


def ising(J: float, h: float) -> SparsePauliOp:
    """Open-chain transverse-field Ising Hamiltonian."""
    return SparsePauliOp.from_sparse_list(
        [("ZZ", [i, i + 1], -J) for i in range(n - 1)] + [("X", [i], -h) for i in range(n)],
        num_qubits=n,
    )


def brickwork(reps: int) -> QuantumCircuit:
    """rx/ry layers and rzz bricks. All angles zero prepares |0...0⟩."""
    return n_local(n, ["rx", "ry"], "rzz", entanglement="pairwise", reps=reps)


def mclachlan_rhs(bowtie: BowtieQGT, which: str, regularization: float = 1e-6):
    """θ̇ = (Re G + λ I)^{-1} b, with b for which="real_time" or "imag_time".

    The small λ handles the redundant directions of G.
    """

    def rhs(_t, theta):
        generalized_qgt, _ = bowtie.evaluate(theta)
        qgt = bowtie.extract_qgt(generalized_qgt)  # real part, as McLachlan requires
        b = bowtie.extract_gradient(generalized_qgt, which=which)
        return np.linalg.solve(qgt + regularization * np.eye(len(b)), b)

    return rhs


def evolve(bowtie: BowtieQGT, which: str, times: np.ndarray) -> np.ndarray:
    """Integrate McLachlan's equation from θ = 0; returns θ at each time, shape (T, N)."""
    solution = solve_ivp(
        mclachlan_rhs(bowtie, which),
        (times[0], times[-1]),
        np.zeros(bowtie.qc.num_parameters),
        t_eval=times,
        rtol=1e-6,
        atol=1e-8,
    )
    return solution.y.T


# ======================================================================================
# 1. REAL-TIME evolution (VarQRTE): |ψ(t)⟩ = e^{-iHt}|0...0⟩
#    Magnetization oscillations after a quench.
# ======================================================================================
hamiltonian = ising(J=0.25, h=1.0)
magnetization = SparsePauliOp.from_sparse_list([("Z", [i], 1 / n) for i in range(n)], n)
ansatz = brickwork(reps=3)
times = np.linspace(0, 6, 61)  # real time t
thetas = evolve(BowtieQGT(ansatz, hamiltonian), "real_time", times)
m_varqrte = [
    Statevector(ansatz.assign_parameters(theta)).expectation_value(magnetization).real
    for theta in thetas
]
# Exact reference: exp(-iHt)|0...0⟩ on the full 2^n state vector.
psi0 = np.zeros(2**n, dtype=complex)
psi0[0] = 1
exact_states = expm_multiply(
    -1j * hamiltonian.to_matrix(sparse=True), psi0, start=times[0], stop=times[-1], num=len(times)
)
m_exact = [Statevector(psi).expectation_value(magnetization).real for psi in exact_states]
print(f"Real time:      max |m_z - exact| = {np.max(np.abs(np.subtract(m_varqrte, m_exact))):.1e}")

# ======================================================================================
# 2. IMAGINARY-TIME evolution (VarQITE): |ψ(τ)⟩ ∝ e^{-Hτ}|+...+⟩
#    Ground-state preparation at the critical point.
# ======================================================================================
hamiltonian = ising(J=1.0, h=1.0)
ansatz = QuantumCircuit(n)
ansatz.h(range(n))  # θ = 0 now prepares |+...+⟩
ansatz.compose(brickwork(reps=4), inplace=True)
bowtie = BowtieQGT(ansatz, hamiltonian)
taus = np.linspace(0, 5, 41)  # imaginary time τ
energies, variances = [], []
for theta in evolve(bowtie, "imag_time", taus):
    generalized_qgt, energy = bowtie.evaluate(theta)
    energies.append(energy)
    variances.append(bowtie.extract_variance(generalized_qgt))
ground_energy = eigsh(hamiltonian.to_matrix(sparse=True), k=1, which="SA")[0][0]
print(f"Imaginary time: E = {energies[-1]:.5f}, exact {ground_energy:.5f}")
print(f"                Var[H] = {variances[-1]:.1e}")

# --- Plots ----------------------------------------------------------------------------
fig, (ax_rt, ax_e, ax_var) = plt.subplots(1, 3, figsize=(13, 3.6), layout="constrained")
ax_rt.plot(times, m_exact, "k-", label="exact")
ax_rt.plot(times, m_varqrte, "o", ms=4, label="VarQRTE")
ax_rt.set(xlabel="real time $t$", ylabel=r"$m_z$", title=r"Real time: quench ($J=0.25,\ h=1$)")
ax_rt.legend()
ax_e.plot(taus, energies, "o-", ms=3, label="VarQITE")
ax_e.axhline(ground_energy, color="k", ls="--", label="ground energy")
ax_e.set(xlabel=r"imaginary time $\tau$", ylabel="E", title=r"Imaginary time: energy ($J=h=1$)")
ax_e.legend()
ax_var.semilogy(taus, variances, "o-", ms=3)
ax_var.set(xlabel=r"imaginary time $\tau$", ylabel="Var[H]", title="Imaginary time: variance")
fig.savefig("mclachlan_ising.png", dpi=150)
