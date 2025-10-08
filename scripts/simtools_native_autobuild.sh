#!/usr/bin/env bash
set -euo pipefail

# Auto-build script for simtools native LightEmission binding.
# Always builds the native extension at container start (unless disabled) to ensure
# the module is up to date with current headers.

if [[ "${SIMTOOLS_NO_AUTO_NATIVE:-}" == "1" ]]; then
  echo "[native-autobuild] Auto build disabled via SIMTOOLS_NO_AUTO_NATIVE=1" >&2
  exec "$@"
fi

if ! command -v cmake >/dev/null 2>&1; then
  echo "[native-autobuild] cmake not installed in image; cannot auto-build native binding" >&2
  exec "$@"
fi

PRIMARY_ROOT="/workdir/external/simtools"
ALT_ROOT="/workdir/external"
# Detect repository root: prefer primary, fall back to alt if it looks like a repo (has pyproject.toml and src/simtools)
if [[ -d "$PRIMARY_ROOT" ]]; then
  SIMTOOLS_ROOT="$PRIMARY_ROOT"
elif [[ -f "$ALT_ROOT/pyproject.toml" && -d "$ALT_ROOT/src/simtools" ]]; then
  SIMTOOLS_ROOT="$ALT_ROOT"
  echo "[native-autobuild] Using alternate root: $SIMTOOLS_ROOT" >&2
else
  echo "[native-autobuild] No simtools source tree found (checked $PRIMARY_ROOT and $ALT_ROOT)" >&2
  exec "$@"
fi

cd "$SIMTOOLS_ROOT"

# Ensure editable src path is discoverable if user hasn't pip-installed yet
export PYTHONPATH="${SIMTOOLS_ROOT}/src:${PRIMARY_ROOT}/src:${ALT_ROOT}/src:${PYTHONPATH:-}"

# Decide if we must build (force) or probe existing
echo "[native-autobuild] Building native extension (always) ..." >&2
mkdir -p build
if [[ "${SIMTOOLS_NATIVE_DEBUG:-}" == "1" ]]; then
  set -x
fi
BUILD_LOG=build/native_build.log
cmake -S . -B build -DSIMTEL_PREFIX="${SIMTEL_PREFIX:-/workdir/sim_telarray/sim_telarray}" \
      -DHESSIO_LIBRARY="${HESSIO_LIBRARY:-/workdir/sim_telarray/hessioxxx/lib/libhessio.so}" \
      -DHESSIO_INCLUDE_DIR="${HESSIO_INCLUDE_DIR:-/workdir/sim_telarray/hessioxxx/include}" ${SIMTOOLS_EXTRA_CMAKE_ARGS:-} \
      ${SIMTOOLS_NATIVE_DEBUG:+-DCMAKE_VERBOSE_MAKEFILE=ON} | tee "$BUILD_LOG" >/dev/null || {
  { set +x || true; } 2>/dev/null
  echo "[native-autobuild] CMake configure failed (see $BUILD_LOG)" >&2; tail -n 80 "$BUILD_LOG" >&2 || true; exec "$@"; }
cmake --build build -j | tee -a "$BUILD_LOG" >/dev/null || {
  { set +x || true; } 2>/dev/null
  echo "[native-autobuild] Build failed (see $BUILD_LOG)" >&2; tail -n 80 "$BUILD_LOG" >&2 || true; exec "$@"; }
{ set +x || true; } 2>/dev/null
echo "[native-autobuild] Listing built artifacts:" >&2
ls -1 build/src/simtools/light_emission 2>/dev/null || true
ls -1 src/simtools/light_emission/*.so 2>/dev/null || true
# If the .so ended up only in build tree, copy it into the package src path for immediate import
shopt -s nullglob
for so in build/src/simtools/light_emission/_le*.so; do
  if [[ -f "$so" ]]; then
    target="src/simtools/light_emission/$(basename $so)"
    if [[ ! -f "$target" ]]; then
      cp "$so" "$target" && echo "[native-autobuild] Copied $(basename $so) to src for import" >&2
    fi
  fi
done
shopt -u nullglob
if command -v ldd >/dev/null 2>&1; then
  for so in src/simtools/light_emission/_le*.so; do
    [ -f "$so" ] && echo "[native-autobuild] ldd $so" >&2 && ldd "$so" | sed 's/^/[native-autobuild]   /' >&2 || true
  done
fi
python - <<'PYVERIFY' || true
import importlib, inspect
try:
    mod = importlib.import_module('simtools.light_emission._le')
    sig = inspect.signature(mod.ff_1m)
    print('[native-autobuild] Post-build signature:', sig)
except Exception as e:  # noqa: BLE001
    print('[native-autobuild] Post-build import failed:', e)
PYVERIFY

exec "$@"
