# Changelog

## 2.0.0 (unreleased)

This release fixes the conventions of the generalized QGT so that every quantity matches
its textbook definition and McLachlan's equation reads `Re(G) θ̇ = b` with no extra factors.
It is **not backwards compatible**; see the migration guide below.

### Breaking changes

- `get_derivatives` is renamed to `evaluate`. It also accepts a plain sequence of values
  ordered like `qc.parameters`. It no longer takes `tracking_time`; timings of the last call
  are always stored in `bowtie.last_timings`.
- `phase_fix` is removed. The connected (gauge-invariant) quantities are always computed.
  With `phase_fix=False` the variance was wrong whenever two bowties acted on disjoint
  qubits, and could even be negative.
- `VarQITE_gradient` is removed from the constructor. Choose the direction when extracting:
  `extract_gradient(generalized_qgt, which="real_time" | "imag_time")`.
- `extract_gradient` now returns McLachlan's right-hand side `b`, so that `Re(G) θ̇ = b`.
  The old output was `-2 b`, and the flag names were swapped (see the table below).
- `extract_qgt` returns the real part by default (the metric McLachlan uses). Use
  `part="complex"` for the previous complex output, or `part="imag"` for the Berry curvature.
- `extract_variance(generalized_qgt)` no longer takes `energy`.
- `accelerator` is replaced by `simulator`, which takes `"CPU"` (default), `"GPU"`, or any
  noiseless statevector `AerSimulator` for full control. The internal `ThreadPoolExecutor`
  is gone. It was never used by Aer, so it had no effect.
- `pbar` and `verbose_init` are merged into `verbosity` (0, 1 or 2).
- `compute_variance` is removed (it had no effect).
- The generalized QGT no longer has the Hamiltonian coefficients folded in. The QGT
  block is unchanged. The old mixed block was `new[i, j] * c_j / 2` (times an extra `i`
  with `VarQITE_gradient=True`), and the old Hamiltonian block was
  `±new[i, j] * c_i * c_j / 4`. Code that indexed into these blocks directly must be
  updated; prefer the `extract_*` methods.
- `remove_idle_qwires` is renamed to `remove_idle_qubits`.
- `evaluate` returns the energy as a `float` for Hermitian Hamiltonians.

### New features

- Identity terms in the Hamiltonian are supported; they shift the energy only. A
  Hamiltonian that is a multiple of the identity raises a "Trivial Hamiltonian" warning.
- `evaluate` accepts a sequence of values or any mapping `assign_parameters` accepts, and
  raises a clear error when a parameter has no value.
- Example: VarQRTE and VarQITE of a transverse-field Ising chain
  (`docs/examples/mclachlan_ising.py`).

### Bug fixes

- Circuits are validated at construction. A parameter used in more than one gate is now an
  error; previously only one gate's derivative was counted and the gradient was silently
  wrong. Parameter expressions and unsupported parameterized gates also give explanatory
  errors.
- Circuits with classical bits, control flow or a parameterized global phase, and
  simulators that are not noiseless statevector simulators, are rejected with a clear
  error instead of crashing later or returning wrong numbers.
- Bowtie circuits are actually transpiled. The result used to be discarded, so gates Aer
  cannot run natively (e.g. `QFTGate`) failed.
- The energy variance is correct when bowties have disjoint support.
- Complex coefficients enter the variance as `conj(c_i) c_j`.

### Packaging

- Runtime dependencies trimmed to `numpy`, `qiskit`, `qiskit-aer` and `tqdm`. Test-only
  dependencies moved to the `dev` extra. The generated `_version.py` is no longer tracked.

### Migrating from 1.0

```python
# 1.0
bowtie = BowtieQGT(qc, obs, phase_fix=True, pbar=1, accelerator="GPU", VarQITE_gradient=False)
gen_qgt, energy = bowtie.get_derivatives(param_dict)
qgt = bowtie.extract_qgt(gen_qgt).real
b = bowtie.extract_gradient(gen_qgt).real / 2  # real time, see table
variance = bowtie.extract_variance(gen_qgt, energy)

# 2.0
bowtie = BowtieQGT(qc, obs, verbosity=1, simulator="GPU")
generalized_qgt, energy = bowtie.evaluate(param_dict)
qgt = bowtie.extract_qgt(generalized_qgt)
b = bowtie.extract_gradient(generalized_qgt, which="real_time")
variance = bowtie.extract_variance(generalized_qgt)
```

How old gradient calls map to the new API:

| 1.0 call | returned | 2.0 equivalent |
|---|---|---|
| `VarQITE_gradient=False`, `extract_gradient(g)` | `-2 b_real` | `-2 * extract_gradient(g, which="real_time")` |
| `VarQITE_gradient=True`, `extract_gradient(g)` | `-2 b_imag = ∇E` | `-2 * extract_gradient(g, which="imag_time")` |

Code that solved `G θ̇ = gradient / 2` with the default `VarQITE_gradient=False` was
integrating `e^{+iHt}`, i.e. backwards in time. Solve `G θ̇ = b` with
`b = extract_gradient(g, which="real_time")` to get `e^{-iHt}`. For a real Hamiltonian
started from a real state, such as a transverse-field Ising quench from `|0...0⟩`, both
directions give the same expectation values, so the old results are unaffected.
