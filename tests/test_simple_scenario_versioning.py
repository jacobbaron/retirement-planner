"""
Tests for simple scenario versioning service.

This module tests the SimpleScenarioVersioningService which stores full copies
of scenarios rather than using base + diff approach.
"""

import pytest

from app.database.models import VersionedScenario
from app.models.scenario import Scenario
from app.models.scenario_versioning import SimpleScenarioVersioningService


@pytest.fixture
def sample_scenario_data():
    """Load the sample scenario fixture data."""
    import json
    from pathlib import Path

    fixture_path = Path(__file__).parent / "fixtures" / "sample_scenario.json"
    with open(fixture_path, "r") as f:
        return json.load(f)


@pytest.fixture
def sample_scenario(sample_scenario_data):
    """Create a sample scenario from test data."""
    return Scenario(**sample_scenario_data)


@pytest.fixture
def versioning_service(db_session):
    """Create a versioning service instance."""
    return SimpleScenarioVersioningService(db_session)


class TestSimpleScenarioVersioning:
    """Test cases for simple scenario versioning."""

    def test_create_version(self, versioning_service, test_user, sample_scenario):
        """Test creating a new scenario version."""
        scenario_id = versioning_service.create_version(
            user_id=test_user.id,
            scenario_data=sample_scenario,
            name="My First Scenario",
            description="Initial version",
        )

        assert scenario_id is not None
        assert scenario_id.startswith("scenario_")
        assert str(test_user.id) in scenario_id

        # Verify it was stored in database
        versioned = (
            versioning_service.db.query(VersionedScenario)
            .filter(VersionedScenario.scenario_id == scenario_id)
            .first()
        )

        assert versioned is not None
        assert versioned.name == "My First Scenario"
        assert versioned.description == "Initial version"
        assert versioned.user_id == test_user.id
        assert versioned.version == "v1"
        assert versioned.parent_version_id is None

    def test_get_scenario(self, versioning_service, test_user, sample_scenario):
        """Test retrieving a scenario version."""
        scenario_id = versioning_service.create_version(
            user_id=test_user.id, scenario_data=sample_scenario, name="Test Scenario"
        )

        retrieved_scenario = versioning_service.get_scenario(scenario_id)

        assert isinstance(retrieved_scenario, Scenario)
        assert (
            retrieved_scenario.scenario_metadata.name
            == sample_scenario.scenario_metadata.name
        )
        assert (
            retrieved_scenario.household.primary_age
            == sample_scenario.household.primary_age
        )
        assert (
            retrieved_scenario.accounts.taxable[0].current_balance
            == sample_scenario.accounts.taxable[0].current_balance
        )

    def test_get_version_info(self, versioning_service, test_user, sample_scenario):
        """Test getting version metadata."""
        scenario_id = versioning_service.create_version(
            user_id=test_user.id,
            scenario_data=sample_scenario,
            name="Test Scenario",
            description="Test description",
        )

        info = versioning_service.get_version_info(scenario_id)

        assert info["scenario_id"] == scenario_id
        assert info["name"] == "Test Scenario"
        assert info["description"] == "Test description"
        assert info["version"] == "v1"
        assert info["parent_version_id"] is None
        assert info["user_id"] == test_user.id
        assert "created_at" in info
        assert "updated_at" in info

    def test_list_user_scenarios(self, versioning_service, test_user, sample_scenario):
        """Test listing all scenarios for a user."""
        # Create multiple scenarios
        scenario_id_1 = versioning_service.create_version(
            user_id=test_user.id, scenario_data=sample_scenario, name="Scenario 1"
        )

        scenario_id_2 = versioning_service.create_version(
            user_id=test_user.id, scenario_data=sample_scenario, name="Scenario 2"
        )

        scenarios = versioning_service.list_user_scenarios(test_user.id)

        assert len(scenarios) == 2
        scenario_names = [s["name"] for s in scenarios]
        assert "Scenario 1" in scenario_names
        assert "Scenario 2" in scenario_names

        # Should be ordered by created_at desc (newest first)
        assert scenarios[0]["name"] == "Scenario 2"
        assert scenarios[1]["name"] == "Scenario 1"

    def test_compare_versions(self, versioning_service, test_user, sample_scenario):
        """Test comparing two scenario versions."""
        # Create first version
        scenario_id_1 = versioning_service.create_version(
            user_id=test_user.id,
            scenario_data=sample_scenario,
            name="Original Scenario",
        )

        # Create modified version
        modified_data = sample_scenario.model_dump()
        modified_data["household"]["primary_age"] = 40
        modified_data["accounts"]["taxable"][0]["current_balance"] = 120000.0
        modified_scenario = Scenario(**modified_data)

        scenario_id_2 = versioning_service.create_version(
            user_id=test_user.id,
            scenario_data=modified_scenario,
            name="Modified Scenario",
        )

        comparison = versioning_service.compare_versions(scenario_id_1, scenario_id_2)

        assert comparison["scenario_1"] == scenario_id_1
        assert comparison["scenario_2"] == scenario_id_2
        assert comparison["has_changes"] is True

        # Check that DeepDiff found the changes
        changes = comparison["changes"]
        assert (
            "values_changed" in changes
            or "dictionary_item_added" in changes
            or "dictionary_item_removed" in changes
        )

    def test_version_history(self, versioning_service, test_user, sample_scenario):
        """Test getting version history."""
        # Create root version
        root_id = versioning_service.create_version(
            user_id=test_user.id, scenario_data=sample_scenario, name="Root Version"
        )

        # Get the versioned scenario to get its ID
        root_versioned = (
            versioning_service.db.query(VersionedScenario)
            .filter(VersionedScenario.scenario_id == root_id)
            .first()
        )

        # Create child version
        child_id = versioning_service.create_version(
            user_id=test_user.id,
            scenario_data=sample_scenario,
            name="Child Version",
            parent_version_id=root_versioned.id,
        )

        # Get the child versioned scenario to get its ID
        child_versioned = (
            versioning_service.db.query(VersionedScenario)
            .filter(VersionedScenario.scenario_id == child_id)
            .first()
        )

        # Create grandchild version
        grandchild_id = versioning_service.create_version(
            user_id=test_user.id,
            scenario_data=sample_scenario,
            name="Grandchild Version",
            parent_version_id=child_versioned.id,
        )

        # Test history from grandchild
        history = versioning_service.get_version_history(grandchild_id)

        assert len(history) == 3
        assert history[0]["name"] == "Root Version"
        assert history[1]["name"] == "Child Version"
        assert history[2]["name"] == "Grandchild Version"

    def test_version_numbering(self, versioning_service, test_user, sample_scenario):
        """Test version numbering logic."""
        # First scenario should be v1
        scenario_id_1 = versioning_service.create_version(
            user_id=test_user.id, scenario_data=sample_scenario, name="First Scenario"
        )

        info_1 = versioning_service.get_version_info(scenario_id_1)
        assert info_1["version"] == "v1"

        # Second scenario should be v2
        scenario_id_2 = versioning_service.create_version(
            user_id=test_user.id, scenario_data=sample_scenario, name="Second Scenario"
        )

        info_2 = versioning_service.get_version_info(scenario_id_2)
        assert info_2["version"] == "v2"

        # Test branching versioning
        versioned_1 = (
            versioning_service.db.query(VersionedScenario)
            .filter(VersionedScenario.scenario_id == scenario_id_1)
            .first()
        )

        branch_id = versioning_service.create_version(
            user_id=test_user.id,
            scenario_data=sample_scenario,
            name="Branch Scenario",
            parent_version_id=versioned_1.id,
        )

        branch_info = versioning_service.get_version_info(branch_id)
        assert branch_info["version"] == "v1.1"

    def test_scenario_not_found(self, versioning_service):
        """Test error handling for non-existent scenarios."""
        with pytest.raises(ValueError, match="Scenario nonexistent not found"):
            versioning_service.get_scenario("nonexistent")

        with pytest.raises(ValueError, match="Scenario nonexistent not found"):
            versioning_service.get_version_info("nonexistent")

        with pytest.raises(ValueError, match="Scenario nonexistent not found"):
            versioning_service.get_version_history("nonexistent")

    def test_immutability_of_stored_data(
        self, versioning_service, test_user, sample_scenario
    ):
        """Test that stored scenario data is independent of original."""
        # Create version
        scenario_id = versioning_service.create_version(
            user_id=test_user.id, scenario_data=sample_scenario, name="Test Scenario"
        )

        # Modify original scenario
        original_age = sample_scenario.household.primary_age
        sample_scenario.household.primary_age = 99

        # Retrieve stored version
        stored_scenario = versioning_service.get_scenario(scenario_id)

        # Stored version should have original value
        assert stored_scenario.household.primary_age == original_age
        assert stored_scenario.household.primary_age != 99

    def test_full_scenario_copy(self, versioning_service, test_user, sample_scenario):
        """Test that full scenario data is copied and stored."""
        scenario_id = versioning_service.create_version(
            user_id=test_user.id, scenario_data=sample_scenario, name="Full Copy Test"
        )

        # Get raw database record
        versioned = (
            versioning_service.db.query(VersionedScenario)
            .filter(VersionedScenario.scenario_id == scenario_id)
            .first()
        )

        # Verify all data is present in JSONB
        stored_data = versioned.scenario_data
        assert "scenario_metadata" in stored_data
        assert "household" in stored_data
        assert "accounts" in stored_data
        assert "expenses" in stored_data
        assert "liabilities" in stored_data
        assert "incomes" in stored_data
        assert "policies" in stored_data
        assert "market_model" in stored_data
        assert "strategy" in stored_data

        # Verify specific values
        assert stored_data["household"]["primary_age"] == 35
        assert stored_data["accounts"]["taxable"][0]["current_balance"] == 50000
        assert stored_data["expenses"]["food"] == 1200

    def test_deepdiff_comparison_accuracy(
        self, versioning_service, test_user, sample_scenario
    ):
        """Test that DeepDiff accurately identifies changes."""
        # Create original
        original_id = versioning_service.create_version(
            user_id=test_user.id, scenario_data=sample_scenario, name="Original"
        )

        # Create modified version with known changes
        modified_data = sample_scenario.model_dump()
        modified_data["household"]["primary_age"] = 40
        modified_data["household"]["spouse_age"] = 38
        modified_data["accounts"]["taxable"][0]["current_balance"] = 120000.0
        modified_scenario = Scenario(**modified_data)

        modified_id = versioning_service.create_version(
            user_id=test_user.id, scenario_data=modified_scenario, name="Modified"
        )

        comparison = versioning_service.compare_versions(original_id, modified_id)

        # Should detect the three changes we made
        changes = comparison["changes"]
        assert comparison["has_changes"] is True

        # DeepDiff should find the specific changes
        if "values_changed" in changes:
            changed_paths = list(changes["values_changed"].keys())
            # Should include our three changes
            assert any("primary_age" in path for path in changed_paths)
            assert any("spouse_age" in path for path in changed_paths)
            assert any("current_balance" in path for path in changed_paths)
