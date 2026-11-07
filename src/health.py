"""System health check and diagnostic module for RetailPulse AI."""

import os
import sys
import platform
from pathlib import Path
from typing import Dict, Any


def get_system_health() -> Dict[str, Any]:
    """Inspect and return current project and runtime health status."""
    project_root = Path(__file__).resolve().parent.parent
    config_file = project_root / "config" / "config.yaml"
    env_example = project_root / ".env.example"

    return {
        "status": "healthy",
        "service": "RetailPulse AI Core",
        "version": "0.1.0",
        "environment": os.getenv("ENVIRONMENT", "development"),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "paths": {
            "project_root": str(project_root),
            "config_exists": config_file.exists(),
            "env_example_exists": env_example.exists(),
        },
    }


if __name__ == "__main__":
    import json

    print(json.dumps(get_system_health(), indent=2))
