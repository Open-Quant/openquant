#!/bin/sh
# Standalone entry: no preinstalled AI-DLC, Python, mise, Node, or Rust.
set -eu
AI_DLC_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
AI_DLC_TARGET=local
AI_DLC_MODE=release
AI_DLC_PLAN=false
# Source mode prepares one environment per checkout. The shared aliases are a
# machine-wide selection, so a second checkout must not claim them implicitly.
AI_DLC_PUBLISH_ALIASES=auto
while [ "$#" -gt 0 ]; do
    case "$1" in
        --source) AI_DLC_MODE=source; shift ;;
        --publish-aliases) AI_DLC_PUBLISH_ALIASES=true; shift ;;
        --target) AI_DLC_TARGET=$2; shift 2 ;;
        --root) AI_DLC_ROOT=$2; shift 2 ;;
        --plan) AI_DLC_PLAN=true; shift ;;
        *) echo "Unknown bootstrap argument: $1" >&2; exit 2 ;;
    esac
done
AI_DLC_PLATFORM="$(uname -s)-$(uname -m)"
# Rosetta must not select Intel artifacts on an Apple silicon machine.
if [ "$(uname -s)" = Darwin ] && [ "$(sysctl -n hw.optional.arm64 2>/dev/null || true)" = 1 ]; then
    AI_DLC_PLATFORM=Darwin-arm64
fi
. "$AI_DLC_ROOT/bootstrap/versions.sh"
. "$AI_DLC_ROOT/bootstrap/download.sh"
AI_DLC_BOOTSTRAP_HOME=${AI_DLC_BOOTSTRAP_HOME:-${XDG_DATA_HOME:-$HOME/.local/share}/ai-dlc/bootstrap}
if [ "$AI_DLC_PLAN" = true ]; then
    printf 'platform=%s\nmode=%s\nuv=%s\npython=%s\nmise=%s\ntarget=%s\npublish_aliases=%s\n' "$AI_DLC_PLATFORM" "$AI_DLC_MODE" "$AI_DLC_UV_URL" "$AI_DLC_PYTHON_VERSION" "$AI_DLC_MISE_URL" "$AI_DLC_TARGET" "$AI_DLC_PUBLISH_ALIASES"
    exit 0
fi
if [ "$AI_DLC_MODE" = release ]; then
    [ -f "$AI_DLC_ROOT/bootstrap/release.sh" ] || { echo 'Release wheel manifest is not available. For AI-DLC development use --source; publish a verified release before distributing project bootstrap.' >&2; exit 1; }
    . "$AI_DLC_ROOT/bootstrap/release.sh"
fi
command -v curl >/dev/null || { echo 'curl and CA certificates are required.' >&2; exit 1; }
mkdir -p "$AI_DLC_BOOTSTRAP_HOME/bin" "$AI_DLC_BOOTSTRAP_HOME/downloads"
AI_DLC_DOWNLOADS="$AI_DLC_BOOTSTRAP_HOME/downloads"
AI_DLC_UV_ARCHIVE="$AI_DLC_DOWNLOADS/uv-$AI_DLC_UV_VERSION-$AI_DLC_UV_TARGET.tar.gz"
if [ ! -f "$AI_DLC_UV_ARCHIVE" ] || [ "$(ai_dlc_hash "$AI_DLC_UV_ARCHIVE")" != "$AI_DLC_UV_SHA256" ]; then
    ai_dlc_download "$AI_DLC_UV_URL" "$AI_DLC_UV_SHA256" "$AI_DLC_UV_ARCHIVE"
fi
AI_DLC_EXTRACT=$(mktemp -d "$AI_DLC_BOOTSTRAP_HOME/bin/.install.XXXXXX")
# Retain staging paths: pathname cleanup could delete an authored replacement.
trap 'printf "Bootstrap retained %s; inspect before removal.\n" "$AI_DLC_EXTRACT" >&2' EXIT
tar -xzf "$AI_DLC_UV_ARCHIVE" -C "$AI_DLC_EXTRACT"
ai_dlc_publish_executable "$AI_DLC_EXTRACT/uv-$AI_DLC_UV_TARGET/uv" "$AI_DLC_BOOTSTRAP_HOME/bin"
ai_dlc_publish_executable "$AI_DLC_EXTRACT/uv-$AI_DLC_UV_TARGET/uvx" "$AI_DLC_BOOTSTRAP_HOME/bin"
export PATH="$AI_DLC_BOOTSTRAP_HOME/bin:$PATH"
export UV_PYTHON_INSTALL_DIR="$AI_DLC_BOOTSTRAP_HOME/python"
export UV_PYTHON_BIN_DIR="$AI_DLC_BOOTSTRAP_HOME/bin"
uv python install "$AI_DLC_PYTHON_VERSION" --managed-python
AI_DLC_ENGINE_PYTHON=$(uv python find --managed-python "$AI_DLC_PYTHON_VERSION")
if [ "$AI_DLC_MODE" = source ]; then
    # uv.lock belongs to this checkout; never execute an older installed release in self CI.
    AI_DLC_SOURCE_KEY=$(printf '%s' "$AI_DLC_ROOT" | cksum | cut -d ' ' -f 1)
    AI_DLC_SOURCE_ENV="$AI_DLC_BOOTSTRAP_HOME/source-$AI_DLC_SOURCE_KEY"
    UV_PROJECT_ENVIRONMENT="$AI_DLC_SOURCE_ENV" uv sync --project "$AI_DLC_ROOT" --locked --python "$AI_DLC_ENGINE_PYTHON"
    AI_DLC_CLI="$AI_DLC_SOURCE_ENV/bin/ai-dlc"
    # The environment records its own checkout so any alias pointing here is attributable.
    printf '%s\n' "$AI_DLC_ROOT" > "$AI_DLC_SOURCE_ENV/ai-dlc-source-root"
else
    ai_dlc_download "$AI_DLC_WHEEL_URL" "$AI_DLC_WHEEL_SHA256" "$AI_DLC_DOWNLOADS/$AI_DLC_WHEEL_NAME"
    ai_dlc_download "$AI_DLC_CONSTRAINTS_URL" "$AI_DLC_CONSTRAINTS_SHA256" "$AI_DLC_DOWNLOADS/constraints.txt"
    uv venv --python "$AI_DLC_ENGINE_PYTHON" "$AI_DLC_BOOTSTRAP_HOME/engine-$AI_DLC_ENGINE_VERSION"
    uv pip install --python "$AI_DLC_BOOTSTRAP_HOME/engine-$AI_DLC_ENGINE_VERSION/bin/python" --require-hashes -r "$AI_DLC_DOWNLOADS/constraints.txt"
    uv pip install --python "$AI_DLC_BOOTSTRAP_HOME/engine-$AI_DLC_ENGINE_VERSION/bin/python" --no-deps "$AI_DLC_DOWNLOADS/$AI_DLC_WHEEL_NAME"
    AI_DLC_CLI="$AI_DLC_BOOTSTRAP_HOME/engine-$AI_DLC_ENGINE_VERSION/bin/ai-dlc"
    # The manifest cannot live inside the wheel it hashes. Keep it beside the engine
    # so project generation can hand generated projects the exact assets that built it.
    cp "$AI_DLC_ROOT/bootstrap/release.sh" "$AI_DLC_BOOTSTRAP_HOME/engine-$AI_DLC_ENGINE_VERSION/release.sh"
fi
AI_DLC_MISE_BINARY="$AI_DLC_DOWNLOADS/mise-$AI_DLC_MISE_VERSION-$AI_DLC_MISE_TARGET"
if [ ! -f "$AI_DLC_MISE_BINARY" ] || [ "$(ai_dlc_hash "$AI_DLC_MISE_BINARY")" != "$AI_DLC_MISE_SHA256" ]; then
    ai_dlc_download "$AI_DLC_MISE_URL" "$AI_DLC_MISE_SHA256" "$AI_DLC_MISE_BINARY"
fi
cp "$AI_DLC_MISE_BINARY" "$AI_DLC_EXTRACT/mise"
ai_dlc_publish_executable "$AI_DLC_EXTRACT/mise" "$AI_DLC_BOOTSTRAP_HOME/bin"
AI_DLC_ALIAS="$AI_DLC_BOOTSTRAP_HOME/bin/ai-dlc"
# Release mode owns the shared aliases. In source mode publish only on request or
# when no working alias exists, so bootstrapping a worktree cannot repoint the
# ai-dlc that every other shell on this machine already selects.
if [ "$AI_DLC_MODE" = release ] || [ "$AI_DLC_PUBLISH_ALIASES" = true ] || [ ! -e "$AI_DLC_ALIAS" ]; then
    ln -sf "$AI_DLC_CLI" "$AI_DLC_ALIAS"
    ln -sf "$AI_DLC_CLI" "$AI_DLC_BOOTSTRAP_HOME/bin/ai-dlc-cli"
    AI_DLC_ALIAS_PUBLISHED=true
else
    AI_DLC_ALIAS_PUBLISHED=false
fi
export PATH="$(dirname "$AI_DLC_CLI"):$PATH"
"$AI_DLC_CLI" project setup --root "$AI_DLC_ROOT" --target "$AI_DLC_TARGET"
if [ -n "${GITHUB_PATH:-}" ]; then
    printf '%s\n%s\n' "$(dirname "$AI_DLC_CLI")" "$AI_DLC_BOOTSTRAP_HOME/bin" >> "$GITHUB_PATH"
fi
printf '\nReady. Add these directories to your PATH for this environment:\n%s\n%s\n' "$(dirname "$AI_DLC_CLI")" "$AI_DLC_BOOTSTRAP_HOME/bin"
if [ "$AI_DLC_ALIAS_PUBLISHED" = true ]; then
    printf 'Global alias %s runs this checkout: %s\n' "$AI_DLC_ALIAS" "$AI_DLC_ROOT"
else
    AI_DLC_ALIAS_ENV=$(dirname "$(dirname "$(readlink "$AI_DLC_ALIAS" 2>/dev/null || printf '%s' "$AI_DLC_ALIAS")")")
    AI_DLC_ALIAS_ROOT=$(cat "$AI_DLC_ALIAS_ENV/ai-dlc-source-root" 2>/dev/null || printf '%s' "$AI_DLC_ALIAS_ENV")
    printf 'Global alias %s is unchanged and runs: %s\n' "$AI_DLC_ALIAS" "$AI_DLC_ALIAS_ROOT"
    printf 'Use %s for this checkout, or rerun with --publish-aliases to repoint the global alias.\n' "$AI_DLC_CLI"
fi
