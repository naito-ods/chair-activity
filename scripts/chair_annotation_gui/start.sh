#!/usr/bin/env bash
set -eu
annotation_app_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$annotation_app_dir/server.py" "$@"
