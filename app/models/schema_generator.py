"""
JSON Schema generator for Pydantic models.

This module provides utilities to generate JSON schemas from Pydantic models
and save them to files for use by other systems.
"""

import json
from pathlib import Path
from typing import Any, Dict

from .scenario import Scenario


def generate_scenario_schema() -> Dict[str, Any]:
    """Generate JSON schema for the Scenario model."""
    return Scenario.model_json_schema()


def save_scenario_schema(output_path: Path) -> None:
    """Save the scenario JSON schema to a file."""
    schema = generate_scenario_schema()

    # Add metadata to the schema
    schema.update(
        {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "$id": "https://retirement-planner.com/schema/scenario_v0_1.json",
            "title": "Retirement Planning Scenario Schema v0.1",
            "description": "Schema for retirement planning scenarios including household, accounts, liabilities, incomes, expenses, policies, market model, and strategy",
        }
    )

    with open(output_path, "w") as f:
        json.dump(schema, f, indent=2)


if __name__ == "__main__":
    # Generate and save the schema when run directly
    schema_path = Path(__file__).parent.parent.parent / "schema" / "scenario_v0_1.json"
    save_scenario_schema(schema_path)
    print(f"Schema saved to {schema_path}")
