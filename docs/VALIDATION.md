# Validation record

Validated on 2026-09-01 from commit `616bece921983ac00387c1452666dfd5669d9fd9` on Windows with Python 3.13.

## Test-package fix

The baseline repository imported shared fixtures as `tests.helpers` but did not mark `tests/` as a package. In an environment that already exposed another top-level `tests` package, collection failed with `ModuleNotFoundError: No module named 'tests.helpers'` in three test modules.

Adding `tests/__init__.py` is the smallest deterministic fix: it does not change application behavior or fixtures; it only makes Python resolve `tests.helpers` to this repository.

## Command and result

```powershell
$env:YIZI_VLM = "0"
$env:YIZI_DETECT = "0"
py -3.13 -m pytest -q
```

Result: 68 passed, with three deprecation warnings (Starlette/httpx compatibility and two Pillow `getdata` warnings).

## UI check

The local server was started with optional visual and detector features disabled. The built-in two-page geometric example was loaded and captured as `docs/screenshots/sample-workspace.png`. No private page, API key, or external writing service was used.
