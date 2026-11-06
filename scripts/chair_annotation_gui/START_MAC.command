#!/bin/bash
set -u
annotation_app_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
for annotation_python in /usr/local/bin/python3 /opt/homebrew/bin/python3 python3; do
  if "$annotation_python" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' >/dev/null 2>&1; then
    "$annotation_python" -X utf8 "$annotation_app_dir/launcher.py" "$@"
    annotation_exit=$?
    if [ "$annotation_exit" -ne 0 ]; then
      read -r -p 'Press Return to close this window. ' annotation_answer
    fi
    exit "$annotation_exit"
  fi
done
printf '%s\n' 'Python 3.10 or newer is required.' 'Install Python from https://www.python.org/downloads/macos/' 'Then open START_MAC.command again.'
read -r -p 'Press Return to close this window. ' annotation_answer
exit 1
