"""Unit tests for system health diagnostics."""

from src.health import get_system_health


def test_system_health_status():
    """Verify system health returns healthy status and required keys."""
    health = get_system_health()
    assert health["status"] == "healthy"
    assert health["version"] == "0.1.0"
    assert "python_version" in health
    assert "platform" in health


def test_system_health_paths():
    """Verify required configuration paths exist."""
    health = get_system_health()
    assert health["paths"]["config_exists"] is True
    assert health["paths"]["env_example_exists"] is True
