# tests/test_virtue_config_sanity.py

from virtues.sanity import validate_virtue_config


def test_virtue_config_is_sane():
    # This will raise if something is structurally broken:
    # - edges referencing non-existent nodes
    # - contexts with zero total weight
    # - derived virtues pointing at unknown nodes
    validate_virtue_config()
