import pytest
import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)
from core.config_manager import ConfigManager


def test_deepcopy_prevents_shared_state():
    cfg1 = ConfigManager()
    cfg2 = ConfigManager()

    # Initially equal
    assert cfg1.config == cfg2.config

    # Modify nested value in cfg1
    cfg1.config['camera']['exposure_ms'] = 123.45

    # cfg2 should remain unchanged
    assert cfg2.config['camera']['exposure_ms'] != 123.45

    # DEFAULT_CONFIG should also remain unchanged
    assert ConfigManager.DEFAULT_CONFIG['camera']['exposure_ms'] != 123.45
