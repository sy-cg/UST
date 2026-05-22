#!/bin/sh

set -eu

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "USTv2 Plot Runner"
echo "Project root: $PROJECT_ROOT"
echo "This wrapper simply calls python run_all_plots.py"
echo "Edit run_all_plots.py if you want to change datasets or params."
echo "============================================================"

python run_all_plots.py
