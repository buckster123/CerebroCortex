#!/usr/bin/env bash
# CerebroCortex installer — checks Python version, installs via pip.
set -euo pipefail

MIN_PYTHON="3.11"

echo "CerebroCortex Installer"
echo "======================="
echo

# Find Python
PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        PYTHON="$cmd"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "ERROR: Python not found. Install Python ${MIN_PYTHON}+ first."
    exit 1
fi

# Check version
PY_VERSION=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$("$PYTHON" -c "import sys; print(sys.version_info.major)")
PY_MINOR=$("$PYTHON" -c "import sys; print(sys.version_info.minor)")

if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 11 ]); then
    echo "ERROR: Python ${MIN_PYTHON}+ required (found ${PY_VERSION})"
    exit 1
fi

echo "Python ${PY_VERSION} found at $(command -v "$PYTHON")"
echo

# Install
echo "Installing cerebro-cortex[all]..."
"$PYTHON" -m pip install 'cerebro-cortex[all]'
echo

# Verify
if command -v cerebro &>/dev/null; then
    echo "Installation successful!"
    echo
    echo "Commands available:"
    echo "  cerebro        — CLI interface"
    echo "  cerebro-mcp    — MCP server (for Claude Code / OpenClaw)"
    echo "  cerebro-api    — REST API + web dashboard"
    echo
    echo "Data directory: ~/.cerebro-cortex/"
    echo "  Override with: export CEREBRO_DATA_DIR=/your/path"
    echo
    echo "Claude Code config (~/.claude.json):"
    echo '  {'
    echo '    "mcpServers": {'
    echo '      "cerebro-cortex": {'
    echo '        "command": "cerebro-mcp"'
    echo '      }'
    echo '    }'
    echo '  }'
else
    echo "Installed, but 'cerebro' not found on PATH."
    echo "You may need to add pip's bin directory to your PATH."
fi
