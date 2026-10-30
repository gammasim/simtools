#!/usr/bin/env bash

# Bootstrap a verified pip before installing packages into the active environment.
# Set SIMTOOLS_PYTHON when the interpreter is not named python.
# Security advisory and bootstrap checksum: https://discuss.python.org/t/109576
set -euo pipefail

python_command=${SIMTOOLS_PYTHON:-python}
pip_wheel="https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl"
pip_sha256="fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70"

# Keep newer stable pip releases; bootstrap older or unrecognized versions.
if ! "$python_command" - <<'PY'
import re
import sys
from importlib.metadata import version

match = re.fullmatch(r"(\d+)\.(\d+)(?:\.(\d+))?", version("pip"))
release = tuple(int(part or 0) for part in match.groups()) if match else ()
sys.exit(0 if release >= (26, 2, 2) else 1)
PY
then
  "$python_command" -m pip install --no-cache-dir --upgrade \
    "pip @ ${pip_wheel}#sha256=${pip_sha256}"
fi

"$python_command" -m pip install "certifi>=2026.10.19" "$@"
