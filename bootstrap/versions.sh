# Generated prerequisite pins. Reviewed official release checksums, 2026-09-02.
AI_DLC_ENGINE_VERSION=0.4.0
AI_DLC_PYTHON_VERSION=3.12.11
AI_DLC_UV_VERSION=0.9.11
AI_DLC_MISE_VERSION=2026.9.1

case "$AI_DLC_PLATFORM" in
  Darwin-arm64)
    AI_DLC_UV_TARGET=aarch64-apple-darwin
    AI_DLC_UV_SHA256=594d9f4cfbd21d5a2f34b0352bf423066a9dab1733c90b5d40e3e227506deb03
    AI_DLC_MISE_TARGET=macos-arm64
    AI_DLC_MISE_SHA256=3cfbe3295dba1a7e43bd02653517a8cc21135ba91f0635b45c98f1ebecc5513f
    ;;
  Darwin-x86_64)
    AI_DLC_UV_TARGET=x86_64-apple-darwin
    AI_DLC_UV_SHA256=14236594b4edbd90929d845766a41a1d4e51d530c9ebbedfb3d93688661f142c
    AI_DLC_MISE_TARGET=macos-x64
    AI_DLC_MISE_SHA256=0718a2aa14a96545a287f77a172d700247bb2d33016e5cf29fce1a05e45ac47a
    ;;
  Linux-aarch64|Linux-arm64)
    AI_DLC_UV_TARGET=aarch64-unknown-linux-gnu
    AI_DLC_UV_SHA256=b695e1796449ea85f967b749f87283678ce284e2c042b4b6fa51fa36ec06f47c
    AI_DLC_MISE_TARGET=linux-arm64
    AI_DLC_MISE_SHA256=0ef0a778eaa8599f3e90a8a0979c9fc3f79922cafb5fa6d39f366d974da33bba
    ;;
  Linux-x86_64)
    AI_DLC_UV_TARGET=x86_64-unknown-linux-gnu
    AI_DLC_UV_SHA256=817c0722b437b4b45b9a7e0231616a09db76bab1b8d178ba7a9680c690db19f0
    AI_DLC_MISE_TARGET=linux-x64
    AI_DLC_MISE_SHA256=c98423c8470d6dc416d9f7036d0646d8ef5ae92ad9186907f8fcc84cbe7db4ea
    ;;
  *) echo "Unsupported bootstrap platform: $AI_DLC_PLATFORM" >&2; exit 1 ;;
esac
AI_DLC_UV_URL="https://github.com/astral-sh/uv/releases/download/$AI_DLC_UV_VERSION/uv-$AI_DLC_UV_TARGET.tar.gz"
AI_DLC_MISE_URL="https://github.com/jdx/mise/releases/download/v$AI_DLC_MISE_VERSION/mise-v$AI_DLC_MISE_VERSION-$AI_DLC_MISE_TARGET"
