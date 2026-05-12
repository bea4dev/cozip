#!/usr/bin/env bash
set -euo pipefail

BIN_DIR="${HOME}/.local/bin"
APP_DIR="${HOME}/.local/share/applications"
SERVICEMENU_DIR="${HOME}/.local/share/kio/servicemenus"
MIME_ROOT="${HOME}/.local/share/mime"
MIME_DIR="${MIME_ROOT}/packages"
DATA_DIR="${HOME}/.local/share/cozip"
ICON_DIR="${DATA_DIR}/icons"

refresh_desktop_caches() {
  command -v update-desktop-database >/dev/null 2>&1 \
    && update-desktop-database "$APP_DIR" 2>/dev/null || true
  command -v update-mime-database >/dev/null 2>&1 \
    && update-mime-database "$MIME_ROOT" 2>/dev/null || true
  command -v kbuildsycoca6 >/dev/null 2>&1 && kbuildsycoca6 >/dev/null 2>&1 || true
  command -v kbuildsycoca5 >/dev/null 2>&1 && kbuildsycoca5 >/dev/null 2>&1 || true
}

remove_file() {
  local path="$1"
  if [[ -e "$path" || -L "$path" ]]; then
    echo "==> Removing $path"
    rm -f "$path"
  fi
}

remove_file "$BIN_DIR/cozip_desktop"
remove_file "$APP_DIR/cozip.desktop"
remove_file "$MIME_DIR/application-x-cozip.xml"
remove_file "$SERVICEMENU_DIR/cozip-compress.desktop"
remove_file "$SERVICEMENU_DIR/cozip-extract.desktop"
remove_file "$SERVICEMENU_DIR/cozip-10-compress-zip.desktop"
remove_file "$SERVICEMENU_DIR/cozip-20-compress-cozip.desktop"
remove_file "$SERVICEMENU_DIR/cozip-30-compress-details.desktop"
remove_file "$SERVICEMENU_DIR/cozip-10-extract-here.desktop"
remove_file "$SERVICEMENU_DIR/cozip-20-extract-details.desktop"
remove_file "$ICON_DIR/comp.ico"
remove_file "$ICON_DIR/decomp.ico"

if [[ -d "$ICON_DIR" ]]; then
  rmdir "$ICON_DIR" 2>/dev/null || true
fi
if [[ -d "$DATA_DIR" ]]; then
  rmdir "$DATA_DIR" 2>/dev/null || true
fi

echo "==> Refreshing desktop caches..."
refresh_desktop_caches

echo ""
echo "Done! CoZip user-local Linux integration was removed."
echo "You may need to restart Dolphin for the service menu change to appear."
