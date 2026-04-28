#!/usr/bin/env python3
"""Test script to verify PDIP solver debug output"""

import sys
sys.path.insert(0, '/workspace/python')

from feedback_stackelberg.config import ExperimentConfig, PDIPConfig
from feedback_stackelberg.experiments import ExperimentRunner

# Configure with limited iterations for quick testing
config = ExperimentConfig()
solver_config = PDIPConfig(max_iter=3, outer_iter=1)

runner = ExperimentRunner(config, solver_config)
result = runner.run('lqr')

print('\n✓ Test completed successfully!')
