# FastChat httpx → httpx2 migration notes

**Branch:** `httpxodus/httpx2-migration` on https://github.com/ProgrammerPlus1998/FastChat
**Base:** `upstream/main` @ 587d5cf (Update constants.py #3733)
**Commit:** 4ea6a03 — `refactor: migrate httpx to httpx2 with dual import`
**Issue:** https://github.com/lm-sys/FastChat/issues/3929 (HTTPXodus, 2026-09-02, OPEN)

## Strategy

Chose **Option A — dual support** (recommended in issue #3929) because:

- FastChat's `requires-python = ">=3.8"` — dropping 3.8/3.9 is out of scope for a
  maintenance-mode project.
- httpx2 requires Python ≥ 3.10.
- httpx is confined to a single call site (`fastchat/serve/openai_api_server.py`'s
  `generate_completion_stream`), so a 4-line `try/except` import block is
  proportionate to the change.

Hard switch (Option B) was considered and rejected for this PR. The issue
text offers both options for the maintainers to choose.

## Diff (2 files, +6/-2)

```diff
diff --git a/fastchat/serve/openai_api_server.py b/fastchat/serve/openai_api_server.py
@@ -20,7 +20,11 @@ from fastapi.exceptions import RequestValidationError
 from fastapi.middleware.cors import CORSMiddleware
 from fastapi.responses import StreamingResponse, JSONResponse
 from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer
-import httpx
+
+try:
+    import httpx2 as httpx
+except ModuleNotFoundError:
+    import httpx

diff --git a/pyproject.toml b/pyproject.toml
@@ -13,7 +13,7 @@ classifiers = [
 dependencies = [
-    "aiohttp", "fastapi", "httpx", "markdown2[all]", "nh3", "numpy",
+    "aiohttp", "fastapi", "httpx", "httpx2>=2.12.0; python_version >= \"3.10\"", "markdown2[all]", "nh3", "numpy",
```

`httpx` stays in runtime deps (so 3.8/3.9 keep working unchanged).
`httpx2` is added as a marker-conditional dep (`python_version >= "3.10"`)
so it's only installed where it can run.
`requires-python` is **not** changed.

## Public API compatibility

The functions used by `generate_completion_stream` exist in both httpx and httpx2
with identical signatures:

- `httpx.AsyncClient(...)`
- `client.stream("POST", url, json=...)`
- `response.aiter_raw()`
- `httpx.Timeout(...)`

No call-site changes are needed beyond the import.

## Validation performed

Setup (Python 3.12.12, venv at `.venv/`):

- `pip install -e .` — pulled in `httpx-0.28.1` and `httpx2-2.12.0` (the
  conditional `python_version >= "3.10"` marker correctly selected httpx2).
- `pip install "black==23.3.0" "pylint==2.8.2" pytest Pillow` (dev extras +
  deps needed by the only nominally-runnable test).

Behavioral checks:

1. **Module import + dual-import resolution:**
   `srv.httpx.__version__ == "2.12.0"` (the `try` block picks httpx2 first).
2. **Fallback path:** uninstalled httpx2, re-imported the module —
   `srv.httpx.__version__ == "0.28.1"` (the `except` branch picks httpx).
   This simulates 3.8/3.9 environments where httpx2 is not installed.
3. **No-import warnings:** `python -W error -c "import fastchat.serve.openai_api_server"`
   succeeds.
4. **black formatting:** the modified file passes `black --check` with the
   pinned `black==23.3.0` from `[project.optional-dependencies].dev`.
   (`format.sh` uses this exact pinned version on the `fastchat/` tree.)

Tests:

- `tests/test_image_utils.py` fails to **collect** because of a pre-existing
  upstream breakage: it imports `resize_image_and_return_image_in_bytes` and
  `image_moderation_filter` from `fastchat.utils`, but those symbols are not
  defined there on current `main`. Verified to fail identically on a clean
  stock checkout (stash + pytest). Unrelated to the httpx migration.
- `tests/test_cli.py`, `tests/test_openai_api.py`, `tests/test_openai_langchain.py`,
  `tests/test_openai_vision_api.py`, and `tests/load_test.py` are all
  **integration tests** that require a live model worker + controller +
  openai-api server, plus external models (vicuna, longchat, chatglm,
  RWKV, ...). These cannot be run in a unit test context. None reference
  `httpx` directly.
- Conclusion: there is no runnable automated test that exercises the
  changed import path. The behavioral checks above are the strongest
  validation possible without spinning up a model worker.

## Risk assessment

- Diff is small (2 files, +6/-2) and the `try/except` form is the standard
  pattern recommended in the HTTPXodus issue template.
- The runtime cost is zero on the success path (one `import` is bound, no
  try/except fires per call).
- The TLS trust-store change in httpx2 does not affect this call site
  because `generate_completion_stream` talks to a local model worker over
  plain HTTP in all FastChat deployments. Worth a line in deployment docs
  if the maintainers later front workers with TLS.
- httpx is left in `dependencies` so existing 3.8/3.9 install paths are
  byte-identical to before.

## Status

- Branch: pushed to `origin/httpxodus/httpx2-migration` (ProgrammerPlus1998 fork)
- PR: **not opened** — waiting on user review per HTTPXodus charter.
- Issue #3929 status: OPEN on lm-sys/FastChat as of 2026-09-02.
