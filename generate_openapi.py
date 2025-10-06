#!/usr/bin/env python3
"""
Script to generate OpenAPI specification for the FastAPI application
"""

import sys
import os
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import the FastAPI app
from src.api.fastapi_anomaly_api import app

# Import FastAPI's utility for generating OpenAPI spec
from fastapi.openapi.utils import get_openapi

def generate_openapi_spec():
    """Generate OpenAPI specification from the FastAPI app"""
    
    # Generate the OpenAPI schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    return openapi_schema

if __name__ == "__main__":
    # Generate the OpenAPI specification
    openapi_spec = generate_openapi_spec()
    
    # Save to file
    import json
    output_path = Path(__file__).parent / "openapi_generated.json"
    with open(output_path, "w") as f:
        json.dump(openapi_spec, f, indent=2)
    
    print(f"OpenAPI specification generated and saved to {output_path}")
    
    # Also save as YAML for better readability
    try:
        import yaml
        yaml_path = Path(__file__).parent / "openapi_generated.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(openapi_spec, f, default_flow_style=False, sort_keys=False)
        print(f"OpenAPI specification also saved as YAML to {yaml_path}")
    except ImportError:
        print("PyYAML not installed. Skipping YAML generation.")
        print("To generate YAML, install PyYAML with: pip install PyYAML")