#!/usr/bin/env bash
# Build the static JupyterLite site into ./dist
#   ./build.sh                 Pyodide runtime loaded from the jsDelivr CDN (small site)
#   ./build.sh --offline URL   bundle a Pyodide tarball (path or URL) so the site needs no CDN
set -euo pipefail
cd "$(dirname "$0")"
[ -d .venv ] || { python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt; }
.venv/bin/python make_notebooks.py
extra=()
if [ "${1:-}" = "--offline" ]; then extra=(--pyodide "${2:?pyodide tarball path or URL}"); fi
.venv/bin/jupyter lite build --contents content --output-dir dist ${extra[@]+"${extra[@]}"}
cp landing.html dist/index.html   # phones -> single-notebook view, larger screens -> JupyterLab
echo "Built ./dist — serve it with:  python3 -m http.server -d dist 8000"
