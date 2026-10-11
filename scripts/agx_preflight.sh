#!/bin/sh
# POSIX launcher: no activation, install, network, sudo, temporary files or host identifiers.
if ! command -v python3 >/dev/null 2>&1; then
  printf '%s\n' '{"status":"partial_inventory","inventory_exit_code":2,"reason":"python3_missing_no_install_attempted","compatibility_verified":false}'
  exit 2
fi
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec python3 -B "$SCRIPT_DIR/agx_preflight.py" "$@"
