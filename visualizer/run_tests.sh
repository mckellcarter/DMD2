#!/bin/bash

# Test runner for DMD2 visualizer

echo "Running unit tests for DMD2 visualizer..."
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Run tests
pytest test_activation_masking.py test_generate_from_activation.py test_process_embeddings.py -v --tb=short

# Check exit code
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✓ All tests passed!${NC}"
else
    echo -e "\n${RED}✗ Some tests failed${NC}"
    exit 1
fi
