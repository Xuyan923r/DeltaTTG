#!/usr/bin/env bash
set -euo pipefail
export DELTATTG_ROOT="${DELTATTG_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${DELTATTG_ROOT}"
exec "${PYTHON_BIN:-python}" -m deltattg.launch "$@"
