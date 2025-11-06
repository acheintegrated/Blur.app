#!/usr/bin/env bash
# scripts/build_venv_core.sh — ARM64-only, Metal wheels, binary-only, CRC-checked
set -Eeuo pipefail

### ── FORCE PYTHON (optional) ────────────────────────────────────────────────
# Allow caller to supply exact interpreter (supports '/usr/bin/arch -arm64 …')
if [[ -n "${PY_BIN:-}" ]]; then
  echo "[build] Using caller-specified PY_BIN: ${PY_BIN}"
  eval "${PY_BIN} -V"
  PY_ARCH="$(eval "${PY_BIN} -c 'import platform; print(platform.machine())'")"
  echo "[build] Python arch=${PY_ARCH}"
  [[ "${PY_ARCH}" == "arm64" ]] || { echo "❌ PY_BIN is not arm64"; exit 1; }
  USE_FORCED_PY=1
else
  USE_FORCED_PY=0
fi

### ── SETTINGS ───────────────────────────────────────────────────────────────
TARGET_ARCH="${1:-arm64}"      # hard default: arm64 only
PY_MINOR="3.11"
REQ_REL="electron/backend/requirements-core.txt"
LLAMA_VERSIONS=("0.3.2" "0.3.1" "0.3.0" "0.2.87")
LLAMA_EXTRA_INDEX="https://abetlen.github.io/llama-cpp-python/whl/metal"
PIP_ONLY_BIN=":all:,!langdetect" # refuse sdists, except for pure-Python langdetect
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REQ_CORE="${ROOT_DIR}/${REQ_REL}"
VENV_DIR="${ROOT_DIR}/blur_env-darwin-${TARGET_ARCH}"
WHEEL_DIR="${ROOT_DIR}/vendor_wheels-${TARGET_ARCH}"

log() { printf "\033[1;36m[build]\033[0m %s\n" "$*"; }
die() { printf "\033[1;31m[fail]\033[0m %s\n" "$*" >&2; exit 1; }

[[ "${TARGET_ARCH}" == "arm64" ]] || die "arm64-only builder. Got TARGET_ARCH='${TARGET_ARCH}'."

### ── PICK PYTHON (ARM-first) ────────────────────────────────────────────────
arm_ok () {
  local p="$1"
  [[ -n "$p" ]] || return 1
  if [[ "$p" != /usr/bin/arch* ]]; then
    [[ -x "${p}" ]] || return 1
  fi
  local arch
  local arch_cmd="$p -c 'import platform;print(platform.machine())' 2>/dev/null || true"
  arch="$(eval "$arch_cmd")"
  [[ "$arch" == "arm64" ]]
}

if [[ "$USE_FORCED_PY" -eq 0 ]]; then
  CANDIDATES=(
    "/usr/bin/arch -arm64 /opt/homebrew/opt/python@${PY_MINOR}/bin/python${PY_MINOR}"
    "/opt/homebrew/opt/python@${PY_MINOR}/bin/python${PY_MINOR}"
    "/usr/bin/arch -arm64 /opt/homebrew/bin/python${PY_MINOR}"
    "/opt/homebrew/bin/python${PY_MINOR}"
    "python${PY_MINOR}"
  )
  PY_BIN=""
  for c in "${CANDIDATES[@]}"; do
    if arm_ok "$c"; then PY_BIN="$c"; break; fi
  done
  [[ -n "$PY_BIN" ]] || die "ARM64 Python ${PY_MINOR} not found (check /opt/homebrew and Rosetta settings)."
  eval "$PY_BIN -V"
  PY_ARCH="$(eval "$PY_BIN -c 'import platform; print(platform.machine())'")"
  echo "[build] Python arch=${PY_ARCH}"
  [[ "$PY_ARCH" == "arm64" ]] || die "picked non-arm64 python (${PY_ARCH})."
fi

### ── RESET ──────────────────────────────────────────────────────────────────
deactivate 2>/dev/null || true
rm -rf "${VENV_DIR}" "${WHEEL_DIR}"
mkdir -p "${WHEEL_DIR}"

### ── VENV ───────────────────────────────────────────────────────────────────
log "Creating venv → ${VENV_DIR}"
eval "$PY_BIN -m venv '${VENV_DIR}'"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install -U pip wheel
log "pip: $(pip --version)"

### ── BINARY-ONLY ENV ────────────────────────────────────────────────────────
export PIP_ONLY_BINARY="${PIP_ONLY_BIN}"
export PIP_NO_BUILD_ISOLATION="1"
export CMAKE_OSX_ARCHITECTURES="arm64"
export ARCHFLAGS="-arch arm64"

### ── INSTALL CORE REQS (WITHOUT LLAMA) ──────────────────────────────────────
[[ -f "${REQ_CORE}" ]] || die "Missing requirements file: ${REQ_CORE}"

TEMP_REQ="$(mktemp)"
grep -v -E '^\s*(llama-cpp-python|langdetect)' "${REQ_CORE}" > "${TEMP_REQ}"

log "Building local wheelhouse (pure-Python will skip)…"
if ! pip wheel -r "${TEMP_REQ}" -w "${WHEEL_DIR}"; then
  log "Wheel build warnings — continuing."
fi

if [ -n "$(ls -A "${WHEEL_DIR}" 2>/dev/null)" ]; then
  log "Installing from wheelhouse…"
  pip install --no-index -f "${WHEEL_DIR}" -r "${TEMP_REQ}"
else
  log "Installing from PyPI (binary-only)…"
  pip install -r "${TEMP_REQ}"
fi

log "Installing langdetect (from source)..."
(unset PIP_ONLY_BINARY; pip install "langdetect==1.0.9")

rm -f "${TEMP_REQ}"

### ── JSONSCHEMA STACK (pure-Python + rpds-py wheel) ─────────────────────────
JSONSCHEMA_PKGS=(
  "jsonschema==4.22.0"
  "jsonschema-specifications==2023.12.1"
  "referencing==0.35.1"
  "rpds-py==0.20.0"
)
log "Building wheelhouse for jsonschema stack…"
if ! pip wheel -w "${WHEEL_DIR}" "${JSONSCHEMA_PKGS[@]}"; then
  log "jsonschema wheel build warnings — continuing."
fi
log "Installing jsonschema stack…"
if [ -n "$(ls -A "${WHEEL_DIR}" 2>/dev/null)" ]; then
  pip install --no-index -f "${WHEEL_DIR}" "${JSONSCHEMA_PKGS[@]}"
else
  pip install "${JSONSCHEMA_PKGS[@]}"
fi

### ── INSTALL LLAMA (METAL WHEELS ONLY, CRC check) ───────────────────────────
install_llama_from_index() {
  local ver="$1"
  log "llama-cpp-python ${ver} via Metal index…"
  local tmp; tmp="$(mktemp -d)"
  pushd "${tmp}" >/dev/null
  if pip download --no-deps --only-binary=":all:" \
      --extra-index-url "${LLAMA_EXTRA_INDEX}" \
      "llama-cpp-python==${ver}"; then
    local whl
    whl="$(ls -1 llama_cpp_python-"${ver}"-*.whl | head -n1 || true)"
    if [[ -n "${whl}" ]]; then
python - "${whl}" <<'PY'
import sys, zipfile
z = zipfile.ZipFile(sys.argv[1])
err = z.testzip()
if err:
    print("zip error at:", err); sys.exit(1)
print("zip ok")
PY
      if [[ $? -ne 0 ]]; then
        echo "[build] bad llama wheel zip: ${whl}"
        popd >/dev/null; rm -rf "${tmp}"; return 1
      fi
      pip install --no-cache-dir "./${whl}" && { popd >/dev/null; rm -rf "${tmp}"; return 0; }
    fi
  fi
  popd >/dev/null
  rm -rf "${tmp}"
  return 1
}

LLAMA_OK=0
for v in "${LLAMA_VERSIONS[@]}"; do
  if install_llama_from_index "${v}"; then LLAMA_OK=1; break; fi
done
[[ "${LLAMA_OK}" == "1" ]] || die "No prebuilt Metal wheel for llama-cpp-python (arm64, py${PY_MINOR}). Refusing to compile."

### ── TRIM + PERMS ───────────────────────────────────────────────────────────
log "Trimming caches & fixing perms…"
find "${VENV_DIR}" -name "__pycache__" -type d -prune -exec rm -rf {} +
find "${VENV_DIR}" -type f -name "*.pyc" -delete
chmod -R a+rX /Users/blur/blur/blur_env-darwin-arm64

log "Trimming caches & fixing perms…"
find "${VENV_DIR}" -name "__pycache__" -type d -prune -exec rm -rf {} +
find "${VENV_DIR}" -type f -name "*.pyc" -delete
chmod -R a+rX "${VENV_DIR}"

log "Removing extended attributes (xattr)..."
xattr -cr "${VENV_DIR}"

### ── VERIFY ARCH OF NATIVES ─────────────────────────────────────────────────
log "Verifying binary architecture…"
count=0
while IFS= read -r -d '' so; do
  ((count++))
  if command -v lipo >/dev/null 2>&1; then
    lipo -info "$so" || file "$so" || true
  else
    file "$so" || true
  fi
  [[ $count -ge 6 ]] && break
done < <(find "${VENV_DIR}" -name "*.so" -print0)

### ── PRINT VERSIONS ─────────────────────────────────────────────────────────
python - <<'PY'
import sys, importlib, platform
mods = [
    "numpy","faiss","llama_cpp","fastapi","uvicorn","orjson","langdetect",
    "jsonschema","referencing","jsonschema_specifications","rpds"
]
print("python:", sys.version)
print("arch:", platform.machine())
for m in mods:
    try:
        mod = importlib.import_module(m.replace("-","_"))
        print(f"{m}: {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"{m}: FAILED - {e}")
PY

log "✅ Venv built successfully for ${TARGET_ARCH} at ${VENV_DIR}"
