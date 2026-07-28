#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

echo "Building Content Localization Blueprint Documentation"
echo "====================================================="

rm -rf build/docs
rm -rf docs/source/_autosummary

# Build proto files
echo "Step 1: Building proto files"
# generate_protos.sh uses bash features (set -o pipefail), so it must be
# invoked with bash rather than sh.
bash protos/generate_protos.sh

echo "Step 2: Mermaid diagrams are rendered directly from docs/source content"

# Build Sphinx documentation
echo "Step 3: Building Sphinx documentation..."
mkdir -p build
# Use sphinx-build from virtual environment if available, otherwise use system sphinx-build
if [ -f ".venv/bin/sphinx-build" ]; then
    .venv/bin/sphinx-build -b html docs/source build/docs/html -w build/sphinx_warnings.txt
else
    sphinx-build -b html docs/source build/docs/html -w build/sphinx_warnings.txt
fi

# Check if documentation was built successfully
if [ ! -d "build/docs/html" ] || [ -z "$(ls -A build/docs/html 2>/dev/null)" ]; then
    echo "Error: Documentation was not built successfully"
    exit 1
fi

# The project requires a warning-free Sphinx build; any collected warning is
# treated as a build failure so drift is caught in CI.
if [ -s "build/sphinx_warnings.txt" ]; then
    echo "Error: Sphinx emitted warnings:"
    cat build/sphinx_warnings.txt
    exit 1
fi

echo ""
echo "Documentation built successfully!"
echo "================================="
echo "HTML documentation: build/docs/html/index.html"
echo "Mermaid sources: docs/source/uml_mermaid/"
echo ""
echo "To view the documentation, open build/docs/html/index.html in your browser"
