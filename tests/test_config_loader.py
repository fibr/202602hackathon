"""Tests for config_loader: YAML loading and deep merge."""

import os
import sys
import tempfile
import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from config_loader import _deep_merge, load_config


class TestDeepMerge:
    def test_simple_override(self):
        base = {'a': 1, 'b': 2}
        override = {'b': 3}
        result = _deep_merge(base, override)
        assert result == {'a': 1, 'b': 3}

    def test_nested_merge(self):
        base = {'robot': {'ip': '1.1.1.1', 'port': 29999}}
        override = {'robot': {'ip': '2.2.2.2'}}
        result = _deep_merge(base, override)
        assert result == {'robot': {'ip': '2.2.2.2', 'port': 29999}}

    def test_add_new_key(self):
        base = {'a': 1}
        override = {'b': 2}
        result = _deep_merge(base, override)
        assert result == {'a': 1, 'b': 2}

    def test_add_nested_key(self):
        base = {'robot': {'ip': '1.1.1.1'}}
        override = {'robot': {'speed': 50}}
        result = _deep_merge(base, override)
        assert result == {'robot': {'ip': '1.1.1.1', 'speed': 50}}

    def test_override_dict_with_scalar(self):
        base = {'a': {'nested': 1}}
        override = {'a': 'replaced'}
        result = _deep_merge(base, override)
        assert result == {'a': 'replaced'}

    def test_override_scalar_with_dict(self):
        base = {'a': 'scalar'}
        override = {'a': {'nested': 1}}
        result = _deep_merge(base, override)
        assert result == {'a': {'nested': 1}}

    def test_empty_override(self):
        base = {'a': 1}
        result = _deep_merge(base, {})
        assert result == {'a': 1}

    def test_deeply_nested(self):
        base = {'l1': {'l2': {'l3': {'value': 'old'}}}}
        override = {'l1': {'l2': {'l3': {'value': 'new'}}}}
        result = _deep_merge(base, override)
        assert result['l1']['l2']['l3']['value'] == 'new'


class TestLoadConfig:
    def test_load_main_config(self):
        """Load the actual robot_config.yaml."""
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'robot_config.yaml')
        config = load_config(config_path)
        assert 'robot' in config
        assert 'camera' in config
        assert config['robot']['ip'] == '192.168.5.1'

    def test_load_with_overrides(self):
        """Test that settings.yaml overrides work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write base config
            base = {'robot': {'ip': '1.1.1.1', 'speed': 30}, 'camera': {'fps': 15}}
            with open(os.path.join(tmpdir, 'robot_config.yaml'), 'w') as f:
                yaml.dump(base, f)

            # Write overrides
            overrides = {'robot': {'speed': 50}}
            with open(os.path.join(tmpdir, 'settings.yaml'), 'w') as f:
                yaml.dump(overrides, f)

            config = load_config(os.path.join(tmpdir, 'robot_config.yaml'))
            assert config['robot']['ip'] == '1.1.1.1'  # unchanged
            assert config['robot']['speed'] == 50       # overridden
            assert config['camera']['fps'] == 15         # unchanged

    def test_load_without_overrides(self):
        """Config loading works when settings.yaml doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = {'robot': {'ip': '1.1.1.1'}}
            with open(os.path.join(tmpdir, 'robot_config.yaml'), 'w') as f:
                yaml.dump(base, f)

            config = load_config(os.path.join(tmpdir, 'robot_config.yaml'))
            assert config['robot']['ip'] == '1.1.1.1'
