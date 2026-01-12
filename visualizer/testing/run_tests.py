#!/usr/bin/env python
"""
Test runner for visualizer tests.

Usage:
    python -m testing.run_tests           # Run all tests
    python -m testing.run_tests adapters  # Run only adapter tests
    python -m testing.run_tests core      # Run only core tests
    python -m testing.run_tests -v        # Verbose output
    python -m testing.run_tests -x        # Stop on first failure
"""

import sys
import subprocess
from pathlib import Path


def main():
    # Determine test directory
    test_dir = Path(__file__).parent

    # Build pytest arguments
    pytest_args = [
        sys.executable, '-m', 'pytest',
        str(test_dir),
        '-v',
        '--tb=short',
    ]

    # Filter by test type if specified
    args = sys.argv[1:]
    test_filter = None

    for arg in args:
        if arg == 'adapters':
            test_filter = 'test_adapters.py'
        elif arg == 'core':
            test_filter = 'test_core_*.py'
        elif arg == 'masking':
            test_filter = 'test_core_masking.py'
        elif arg == 'generator':
            test_filter = 'test_core_generator.py'
        elif arg.startswith('-'):
            pytest_args.append(arg)

    if test_filter:
        pytest_args[3] = str(test_dir / test_filter)

    # Add coverage if available
    try:
        import pytest_cov
        pytest_args.extend(['--cov=adapters', '--cov=core', '--cov-report=term-missing'])
    except ImportError:
        pass

    print(f"Running: {' '.join(pytest_args)}")
    result = subprocess.run(pytest_args)
    sys.exit(result.returncode)


if __name__ == '__main__':
    main()
