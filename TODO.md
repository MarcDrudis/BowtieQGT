# TODO

Deferred items from the October 2026 review. Delete entries as they are done.

## Docs (for a later docs PR)

- Sphinx: every API object is documented twice (`automodule :members:` plus explicit
  `autoclass`/`autofunction` in `docs/source/api/*.rst`), giving ~25 duplicate-object
  warnings. Keep one of the two.
- Docstrings in `bowtie_circuits.py` (`parameter_bowtie`, `observable_bowtie`) and
  `bowtieqgt.py` (`get_slice`, `sparse_overlap_tensors`) use bare `|ψ⟩` and unindented
  lists, which docutils misparses. Wrap the kets in double backticks.
- `docs/source/conf.py`: the intersphinx Qiskit URL (`docs.quantum.ibm.com`) is stale;
  `html_static_path` points at `_static`, which now exists but only holds the example
  plot.
- `docs/source/installation.rst` mentions the Furo theme (the config uses
  `sphinx_nefertiti`), a `uv.lock` and a `.pre-commit-config.yaml` that do not exist,
  and "Batch size" performance tips that do not apply.
- `docs/GENERATING_API_DOCS.md`: `uv pip install -e ".[dev]"` does not install Sphinx,
  because the Sphinx deps are in `[dependency-groups]`, not in the `dev` extra.
- Consider `myst-parser` so `docs/source/changelog.rst` can include `CHANGELOG.md`
  instead of linking to it.
- Run `docs/examples/mclachlan_ising.py` in the docs build (sphinx-gallery or similar)
  so the plot cannot go stale. Today the PNG is committed by hand.

## Tooling

- `pyproject.toml` configures towncrier with `changelog.d/`, but the directory does not
  exist. Either create it and write fragments, or drop towncrier and edit
  `CHANGELOG.md` directly.
- Add a `.pre-commit-config.yaml` (the docs describe one).
- Releases: `V1.0.0` and `v1.0` both point to commits that are not on `main`, so
  `setuptools_scm` reports `0.1.devN`. Tag the release commit on `main` as `v2.0.0`.
  Keep one tag style.
