# Changelog

## [2.0.0] - 2026-04-13

### Changed
- Complete architectural restructuring to support library-first usage.
- Python 3.11+ required.
- `setup.py` and `requirements.txt` replaced with `pyproject.toml` and Hatchling build backend.
- Extracted I/O logic out of `EventEmulator` into dedicated `v2ecore/output/` modules.
- `EventEmulator.generate_events` is now a pure mathematical function and doesn't write to file handles.
- Heavy dependencies `numba` and `dv-processing` are now optional (`pip install v2e[numba,aedat4]`).
- Cleaned up naming conventions according to PEP-8.
- Added type annotations across public API and enabled `mypy --strict` checking via pre-commit and CI.
- Formatted and linted codebase using `ruff`.
- `v2e` main entry point significantly thinned via new `V2EPipeline`.
- Moved legacy `dataset_scripts` to `examples/` and isolated synthetic source code inside `v2ecore/synthetic/`.
- Introduced `pytest` test suite setup.
