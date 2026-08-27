#!/bin/bash
set -e

# Ensure we are in the script's directory
cd "$(dirname "$0")"

# Output directory for the executable
DIST_DIR="./resources/backend"
mkdir -p "$DIST_DIR"

# Clean previous builds
rm -rf build "$DIST_DIR"

echo "Building crec-service..."

BACKEND_LIB_DIR="$(pwd)/backend_lib"
export PYTHONPATH=$PYTHONPATH:$(pwd)/..:$BACKEND_LIB_DIR

# PyInstaller command in one block to avoid line continuation issues
pyinstaller --noconfirm --onedir --clean \
    --name "crec-service" \
    --distpath "$DIST_DIR" \
    --paths "$BACKEND_LIB_DIR" \
    --collect-all "crec" \
    --collect-all "pynput" \
    --collect-all "mss" \
    --hidden-import "pynput.keyboard._darwin" \
    --hidden-import "pynput.mouse._darwin" \
    --hidden-import "crec.observers.screen" \
    --hidden-import "parse_raw_trace" \
    --hidden-import "backports.tarfile" \
    --hidden-import "AVFoundation" \
    --hidden-import "CoreMedia" \
    --additional-hooks-dir "./resources/hooks" \
    --exclude-module "numpy" \
    --exclude-module "shapely" \
    --exclude-module "pydantic" \
    --exclude-module "pydantic_core" \
    --exclude-module "setuptools" \
    --exclude-module "pkg_resources" \
    --exclude-module "docutils" \
    --exclude-module "sqlalchemy" \
    --exclude-module "sklearn" \
    --exclude-module "pandas" \
    --exclude-module "scipy" \
    --exclude-module "torch" \
    --exclude-module "matplotlib" \
    --exclude-module "pydrive" \
    --exclude-module "astropy" \
    --exclude-module "bokeh" \
    --exclude-module "black" \
    --exclude-module "botocore" \
    --exclude-module "comm" \
    --exclude-module "cv2" \
    --exclude-module "datashader" \
    --exclude-module "datasets" \
    --exclude-module "glm_ocr" \
    --exclude-module "grpc" \
    --exclude-module "grpc_tools" \
    --exclude-module "holoviews" \
    --exclude-module "huggingface_hub" \
    --exclude-module "hvplot" \
    --exclude-module "imageio" \
    --exclude-module "IPython" \
    --exclude-module "ipykernel" \
    --exclude-module "ipywidgets" \
    --exclude-module "jedi" \
    --exclude-module "jsonschema" \
    --exclude-module "jupyter_client" \
    --exclude-module "litellm" \
    --exclude-module "mcp" \
    --exclude-module "nbformat" \
    --exclude-module "nltk" \
    --exclude-module "onnxruntime" \
    --exclude-module "openai" \
    --exclude-module "opentelemetry" \
    --exclude-module "panel" \
    --exclude-module "plotly" \
    --exclude-module "polars" \
    --exclude-module "pyarrow" \
    --exclude-module "PyQt5" \
    --exclude-module "pytest" \
    --exclude-module "selenium" \
    --exclude-module "sentry_sdk" \
    --exclude-module "setuptools.tests" \
    --exclude-module "setuptools._distutils.tests" \
    --exclude-module "skimage" \
    --exclude-module "sphinx" \
    --exclude-module "test" \
    --exclude-module "tiktoken" \
    --exclude-module "timm" \
    --exclude-module "tokenizers" \
    --exclude-module "traitlets" \
    --exclude-module "transformers" \
    --exclude-module "uvicorn" \
    --exclude-module "zmq" \
    launcher.py

# Developer ID used for code signing. Signing lets the helper inherit
# Accessibility and Screen Recording permissions from the parent Electron app,
# so the user grants them once instead of twice. Set it to your own identity:
#
#   export CREC_SIGNING_IDENTITY="Developer ID Application: Your Name (TEAMID)"
#
# List available identities with: security find-identity -v -p codesigning
# scripts/afterPack.js reads the same variable, so export it once.
SIGNING_IDENTITY="${CREC_SIGNING_IDENTITY:-}"

if [ -z "$SIGNING_IDENTITY" ]; then
    echo ""
    echo "CREC_SIGNING_IDENTITY is not set, leaving crec-service unsigned."
    echo "The app will still run, but macOS will prompt for Accessibility and"
    echo "Screen Recording permissions separately for the helper process."
    echo ""
    echo "Build complete. Executable is at $DIST_DIR/crec-service/crec-service"
    exit 0
fi

echo "Signing crec-service and all internal binaries as: $SIGNING_IDENTITY"

# Sign all .so and .dylib files in _internal first with Developer ID
find "$DIST_DIR/crec-service/_internal" -name "*.so" -o -name "*.dylib" | while read f; do
    codesign --force --sign "$SIGNING_IDENTITY" "$f" 2>/dev/null || true
done

# Sign the main executable with Developer ID and inherit entitlements
# This allows the binary to inherit TCC permissions from the parent Electron app
codesign --force --options runtime --deep --sign "$SIGNING_IDENTITY" --entitlements ./resources/entitlements.inherit.plist "$DIST_DIR/crec-service/crec-service"

echo "Build complete. Executable is at $DIST_DIR/crec-service/crec-service"
echo ""
echo "Backend signed with Developer ID - TCC permissions will inherit from parent app."
