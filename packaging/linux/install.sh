#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

BIN_DIR="${HOME}/.local/bin"
APP_DIR="${HOME}/.local/share/applications"
SERVICEMENU_DIR="${HOME}/.local/share/kio/servicemenus"
MIME_DIR="${HOME}/.local/share/mime/packages"

echo "==> Building cozip_desktop (release)..."
cargo build -p cozip_desktop --release --quiet --manifest-path "$REPO_ROOT/Cargo.toml"

echo "==> Installing binary..."
mkdir -p "$BIN_DIR"
cp "$REPO_ROOT/target/release/cozip_desktop" "$BIN_DIR/cozip_desktop"
chmod +x "$BIN_DIR/cozip_desktop"

echo "==> Installing desktop entry..."
mkdir -p "$APP_DIR"
cp "$SCRIPT_DIR/cozip.desktop" "$APP_DIR/cozip.desktop"
update-desktop-database "$APP_DIR" 2>/dev/null || true

echo "==> Registering MIME type..."
mkdir -p "$MIME_DIR"
cp "$SCRIPT_DIR/mime/application-x-cozip.xml" "$MIME_DIR/application-x-cozip.xml"
update-mime-database "${HOME}/.local/share/mime" 2>/dev/null || true

echo "==> Installing Dolphin service menus..."
mkdir -p "$SERVICEMENU_DIR"
cp "$SCRIPT_DIR/cozip-compress-servicemenu.desktop" "$SERVICEMENU_DIR/cozip-compress.desktop"
cp "$SCRIPT_DIR/cozip-extract-servicemenu.desktop"  "$SERVICEMENU_DIR/cozip-extract.desktop"
chmod +x "$SERVICEMENU_DIR/cozip-compress.desktop" "$SERVICEMENU_DIR/cozip-extract.desktop"

echo ""
echo "Done! Make sure $BIN_DIR is in your PATH."
echo "You may need to restart Dolphin (or log out/in) for the service menus to appear."
