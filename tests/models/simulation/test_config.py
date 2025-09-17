"""
Tests for simulation configuration model.

This module tests the SimulationConfig Pydantic model including:
- Configuration validation
- Provider compatibility checks
- Edge case handling
- Helper method functionality
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from app.models.simulation.config import SimulationConfig
from app.models.time_grid import InflationAdjuster, TimeGrid, UnitSystem
from tests.models.simulation.test_protocols import (
    MockExpenseProvider,
    MockIncomeProvider,
    MockLiabilityProvider,
    MockPortfolioEngine,
    MockRebalancingStrategy,
    MockReturnsProvider,
    MockTaxCalculator,
    MockWithdrawalPolicy,
)


@pytest.fixture
def unit_system():
    """Create a unit system for testing."""
    time_grid = TimeGrid(start_year=2024, end_year=2054, base_year=2024)
    inflation_adjuster = InflationAdjuster(inflation_rate=0.03, base_year=2024)
    return UnitSystem(
        time_grid=time_grid, inflation_adjuster=inflation_adjuster, display_mode="real"
    )


@pytest.fixture
def mock_providers():
    """Create a set of mock providers for testing."""
    return {
        "returns": MockReturnsProvider(num_assets=2),
        "income": MockIncomeProvider(base_income=50000),
        "expenses": MockExpenseProvider(base_expenses=40000),
        "liabilities": MockLiabilityProvider(annual_payment=12000),
        "withdrawal": MockWithdrawalPolicy(withdrawal_rate=0.04),
        "rebalancing": MockRebalancingStrategy(threshold=0.05),
        "taxes": MockTaxCalculator(tax_rate=0.22),
        "portfolio": MockPortfolioEngine(),
    }


@pytest.fixture
def valid_config_data(unit_system, mock_providers):
    """Create valid configuration data for testing."""
    return {
        "unit_system": unit_system,
        "num_paths": 1000,
        "initial_portfolio_balance": 500000.0,
        "target_asset_weights": {"asset_0": 0.6, "asset_1": 0.4},
        "returns_provider": mock_providers["returns"],
        "withdrawal_policy": mock_providers["withdrawal"],
        "portfolio_engine": mock_providers["portfolio"],
        "rebalancing_strategy": mock_providers["rebalancing"],
        "income_provider": mock_providers["income"],
        "expense_provider": mock_providers["expenses"],
        "liability_provider": mock_providers["liabilities"],
        "tax_calculator": mock_providers["taxes"],
    }


class TestSimulationConfigValidation:
    """Test basic SimulationConfig validation."""

    def test_valid_config_creation(self, valid_config_data):
        """Test creating a valid configuration."""
        config = SimulationConfig(**valid_config_data)

        assert config.num_paths == 1000
        assert config.initial_portfolio_balance == 500000.0
        assert config.target_asset_weights == {"asset_0": 0.6, "asset_1": 0.4}
        assert config.income_provider is not None
        assert config.expense_provider is not None

    def test_minimal_config_creation(self, unit_system, mock_providers):
        """Test creating a minimal configuration with only required fields."""
        config = SimulationConfig(
            unit_system=unit_system,
            num_paths=100,
            initial_portfolio_balance=100000.0,
            target_asset_weights={
                "asset_0": 0.7,
                "asset_1": 0.3,
            },  # Use matching asset names
            returns_provider=mock_providers["returns"],
            withdrawal_policy=mock_providers["withdrawal"],
            portfolio_engine=mock_providers["portfolio"],
            rebalancing_strategy=mock_providers["rebalancing"],
        )

        assert config.income_provider is None
        assert config.expense_provider is None
        assert config.liability_provider is None
        assert config.tax_calculator is None

    def test_invalid_num_paths(self, valid_config_data):
        """Test validation of num_paths field."""
        # Test zero paths
        valid_config_data["num_paths"] = 0
        with pytest.raises(ValueError, match="greater than 0"):
            SimulationConfig(**valid_config_data)

        # Test negative paths
        valid_config_data["num_paths"] = -100
        with pytest.raises(ValueError, match="greater than 0"):
            SimulationConfig(**valid_config_data)

        # Test too many paths
        valid_config_data["num_paths"] = 200000
        with pytest.raises(ValueError, match="less than or equal to 100000"):
            SimulationConfig(**valid_config_data)

    def test_invalid_initial_balance(self, valid_config_data):
        """Test validation of initial portfolio balance."""
        valid_config_data["initial_portfolio_balance"] = -1000
        with pytest.raises(ValueError, match="greater than or equal to 0"):
            SimulationConfig(**valid_config_data)

    def test_invalid_random_seed(self, valid_config_data):
        """Test validation of random seed."""
        valid_config_data["random_seed"] = -1
        with pytest.raises(ValueError, match="greater than or equal to 0"):
            SimulationConfig(**valid_config_data)

    def test_invalid_cash_buffer(self, valid_config_data):
        """Test validation of cash buffer months."""
        # Test negative buffer
        valid_config_data["cash_buffer_months"] = -1
        with pytest.raises(ValueError, match="greater than or equal to 0"):
            SimulationConfig(**valid_config_data)

        # Test too large buffer
        valid_config_data["cash_buffer_months"] = 120
        with pytest.raises(ValueError, match="less than or equal to 60"):
            SimulationConfig(**valid_config_data)

    def test_invalid_penalty_rate(self, valid_config_data):
        """Test validation of emergency withdrawal penalty."""
        # Test negative penalty
        valid_config_data["emergency_withdrawal_penalty"] = -0.1
        with pytest.raises(ValueError, match="greater than or equal to 0"):
            SimulationConfig(**valid_config_data)

        # Test penalty > 100%
        valid_config_data["emergency_withdrawal_penalty"] = 1.5
        with pytest.raises(ValueError, match="less than or equal to 1"):
            SimulationConfig(**valid_config_data)


class TestAssetWeightValidation:
    """Test asset weight validation."""

    def test_valid_asset_weights(self, valid_config_data):
        """Test various valid asset weight configurations."""
        # Two assets (already properly configured)
        config = SimulationConfig(**valid_config_data)
        assert config.target_asset_weights == {"asset_0": 0.6, "asset_1": 0.4}

        # Three assets
        valid_config_data["target_asset_weights"] = {
            "asset_0": 0.5,
            "asset_1": 0.3,
            "asset_2": 0.2,
        }
        # Need to update mock provider for 3 assets
        valid_config_data["returns_provider"] = MockReturnsProvider(num_assets=3)
        config = SimulationConfig(**valid_config_data)
        assert len(config.target_asset_weights) == 3

    def test_empty_asset_weights(self, valid_config_data):
        """Test empty asset weights."""
        valid_config_data["target_asset_weights"] = {}
        with pytest.raises(ValueError, match="target_asset_weights cannot be empty"):
            SimulationConfig(**valid_config_data)

    def test_negative_asset_weights(self, valid_config_data):
        """Test negative asset weights."""
        valid_config_data["target_asset_weights"] = {"asset_0": 0.8, "asset_1": -0.2}
        with pytest.raises(ValueError, match="Weight for asset_1 cannot be negative"):
            SimulationConfig(**valid_config_data)

    def test_asset_weights_exceed_one(self, valid_config_data):
        """Test asset weights exceeding 1.0."""
        valid_config_data["target_asset_weights"] = {"asset_0": 1.2, "asset_1": 0.3}
        with pytest.raises(ValueError, match="Weight for asset_0 cannot exceed 1.0"):
            SimulationConfig(**valid_config_data)

    def test_asset_weights_dont_sum_to_one(self, valid_config_data):
        """Test asset weights that don't sum to 1.0."""
        # Weights sum to 0.8
        valid_config_data["target_asset_weights"] = {"asset_0": 0.5, "asset_1": 0.3}
        with pytest.raises(ValueError, match="Asset weights must sum to 1.0"):
            SimulationConfig(**valid_config_data)

        # Weights sum to 1.2
        valid_config_data["target_asset_weights"] = {"asset_0": 0.7, "asset_1": 0.5}
        with pytest.raises(ValueError, match="Asset weights must sum to 1.0"):
            SimulationConfig(**valid_config_data)

    def test_asset_weights_floating_point_tolerance(self, valid_config_data):
        """Test that small floating point errors are tolerated."""
        # Slightly less than 1.0 (within tolerance)
        valid_config_data["target_asset_weights"] = {"asset_0": 0.6, "asset_1": 0.399}
        config = SimulationConfig(**valid_config_data)
        assert config.target_asset_weights == {"asset_0": 0.6, "asset_1": 0.399}

        # Slightly more than 1.0 (within tolerance)
        valid_config_data["target_asset_weights"] = {"asset_0": 0.6, "asset_1": 0.401}
        config = SimulationConfig(**valid_config_data)
        assert config.target_asset_weights == {"asset_0": 0.6, "asset_1": 0.401}

    def test_non_numeric_asset_weights(self, valid_config_data):
        """Test non-numeric asset weights."""
        valid_config_data["target_asset_weights"] = {
            "asset_0": "not_a_number",
            "asset_1": 0.4,
        }
        with pytest.raises(ValueError, match="Input should be a valid number"):
            SimulationConfig(**valid_config_data)


class TestProviderCompatibility:
    """Test provider compatibility validation."""

    def test_asset_name_mismatch(self, valid_config_data):
        """Test mismatch between provider assets and config weights."""
        # Provider has asset_0, asset_1 but config has stocks, bonds
        valid_config_data["target_asset_weights"] = {"stocks": 0.6, "bonds": 0.4}

        with pytest.raises(ValueError, match="Asset mismatch between returns provider"):
            SimulationConfig(**valid_config_data)

    def test_compatible_asset_names(self, valid_config_data):
        """Test compatible asset names between provider and config."""
        # Both provider and config use asset_0, asset_1
        valid_config_data["target_asset_weights"] = {"asset_0": 0.6, "asset_1": 0.4}

        config = SimulationConfig(**valid_config_data)
        assert config.get_asset_names() == ["asset_0", "asset_1"]


class TestSimulationFeasibility:
    """Test simulation feasibility validation."""

    def test_zero_balance_no_income_fails(self, valid_config_data):
        """Test that zero initial balance with no income fails."""
        valid_config_data["initial_portfolio_balance"] = 0
        valid_config_data["income_provider"] = None

        with pytest.raises(
            ValueError,
            match="Cannot run simulation with zero initial balance and no income source",
        ):
            SimulationConfig(**valid_config_data)

    def test_zero_balance_with_income_succeeds(self, valid_config_data):
        """Test that zero initial balance with income succeeds."""
        valid_config_data["initial_portfolio_balance"] = 0
        # Keep income_provider

        config = SimulationConfig(**valid_config_data)
        assert config.initial_portfolio_balance == 0
        assert config.income_provider is not None

    def test_cash_buffer_without_expenses_fails(self, valid_config_data):
        """Test that cash buffer without expense provider fails."""
        valid_config_data["cash_buffer_months"] = 6
        valid_config_data["expense_provider"] = None

        with pytest.raises(
            ValueError, match="Cannot set cash buffer without expense provider"
        ):
            SimulationConfig(**valid_config_data)

    def test_very_long_simulation_fails(self, valid_config_data):
        """Test that very long simulations fail."""
        # Create a simulation that exceeds 100 years by using a large range
        # Since TimeGrid limits end_year to 2100, we need to use an early start year
        time_grid = TimeGrid(
            start_year=1999, end_year=2100, base_year=2024
        )  # 102 years
        inflation_adjuster = InflationAdjuster(inflation_rate=0.03, base_year=2024)
        long_unit_system = UnitSystem(
            time_grid=time_grid,
            inflation_adjuster=inflation_adjuster,
            display_mode="real",
        )
        valid_config_data["unit_system"] = long_unit_system

        with pytest.raises(ValueError, match="Simulation time horizon too long"):
            SimulationConfig(**valid_config_data)

    def test_large_simulation_warning(self, valid_config_data):
        """Test warning for large simulations."""
        # Create a large simulation (60 years × 60k paths)
        time_grid = TimeGrid(start_year=2024, end_year=2084, base_year=2024)
        inflation_adjuster = InflationAdjuster(inflation_rate=0.03, base_year=2024)
        large_unit_system = UnitSystem(
            time_grid=time_grid,
            inflation_adjuster=inflation_adjuster,
            display_mode="real",
        )
        valid_config_data["unit_system"] = large_unit_system
        valid_config_data["num_paths"] = 60000

        with pytest.raises(ValueError, match="Large simulation detected"):
            SimulationConfig(**valid_config_data)


class TestConfigHelperMethods:
    """Test SimulationConfig helper methods."""

    def test_get_simulation_years(self, valid_config_data):
        """Test getting simulation years."""
        config = SimulationConfig(**valid_config_data)
        assert config.get_simulation_years() == 31  # 2024-2054 inclusive

    def test_get_asset_names(self, valid_config_data):
        """Test getting asset names."""
        config = SimulationConfig(**valid_config_data)
        assert config.get_asset_names() == ["asset_0", "asset_1"]

    def test_get_target_weights_array(self, valid_config_data):
        """Test getting target weights as numpy array."""
        config = SimulationConfig(**valid_config_data)
        weights = config.get_target_weights_array()

        assert isinstance(weights, np.ndarray)
        assert weights.shape == (2,)
        assert np.allclose(weights, [0.6, 0.4])

    def test_has_cash_flows(self, valid_config_data):
        """Test checking for cash flow providers."""
        # With all providers
        config = SimulationConfig(**valid_config_data)
        assert config.has_cash_flows() is True

        # Without any cash flow providers
        valid_config_data["income_provider"] = None
        valid_config_data["expense_provider"] = None
        valid_config_data["liability_provider"] = None
        config = SimulationConfig(**valid_config_data)
        assert config.has_cash_flows() is False

    def test_has_taxes(self, valid_config_data):
        """Test checking for tax calculator."""
        # With tax calculator
        config = SimulationConfig(**valid_config_data)
        assert config.has_taxes() is True

        # Without tax calculator
        valid_config_data["tax_calculator"] = None
        config = SimulationConfig(**valid_config_data)
        assert config.has_taxes() is False

    def test_get_provider_summary(self, valid_config_data):
        """Test getting provider summary."""
        config = SimulationConfig(**valid_config_data)
        summary = config.get_provider_summary()

        expected_providers = {
            "returns",
            "withdrawal",
            "portfolio",
            "rebalancing",
            "income",
            "expenses",
            "liabilities",
            "taxes",
        }
        assert set(summary.keys()) == expected_providers
        assert summary["returns"] == "MockReturnsProvider"
        assert summary["income"] == "MockIncomeProvider"

    def test_create_simulation_summary(self, valid_config_data):
        """Test creating simulation summary."""
        valid_config_data["random_seed"] = 42
        config = SimulationConfig(**valid_config_data)
        summary = config.create_simulation_summary()

        assert summary["simulation_years"] == 31
        assert summary["num_paths"] == 1000
        assert summary["initial_balance"] == 500000.0
        assert summary["asset_allocation"] == {"asset_0": 0.6, "asset_1": 0.4}
        assert summary["has_cash_flows"] is True
        assert summary["has_taxes"] is True
        assert summary["random_seed"] == 42
        assert "time_period" in summary
        assert summary["time_period"]["start_year"] == 2024


class TestValidateForSimulation:
    """Test runtime validation before simulation."""

    def test_valid_providers_pass_validation(self, valid_config_data):
        """Test that valid providers pass runtime validation."""
        config = SimulationConfig(**valid_config_data)
        # Should not raise an exception
        config.validate_for_simulation()

    def test_invalid_returns_provider_fails_validation(self, valid_config_data):
        """Test that invalid returns provider fails validation."""
        # Create a mock that raises an exception
        bad_provider = MagicMock()
        bad_provider.generate_returns.side_effect = RuntimeError("Provider failed")
        bad_provider.get_asset_names.return_value = [
            "asset_0",
            "asset_1",
        ]  # Match config assets
        valid_config_data["returns_provider"] = bad_provider

        config = SimulationConfig(**valid_config_data)
        with pytest.raises(ValueError, match="Returns provider validation failed"):
            config.validate_for_simulation()

    def test_returns_provider_wrong_shape_fails(self, valid_config_data):
        """Test that returns provider with wrong shape fails."""
        # Create a mock that returns wrong shape
        bad_provider = MagicMock()
        bad_provider.generate_returns.return_value = np.zeros(
            (3, 1, 1)
        )  # Wrong number of assets
        bad_provider.get_asset_names.return_value = [
            "asset_0",
            "asset_1",
        ]  # Match config assets
        valid_config_data["returns_provider"] = bad_provider

        config = SimulationConfig(**valid_config_data)
        with pytest.raises(ValueError, match="Returns provider shape mismatch"):
            config.validate_for_simulation()

    def test_portfolio_engine_missing_method_fails(self, valid_config_data):
        """Test that portfolio engine missing initialize method fails."""
        # Create a mock without initialize method
        bad_engine = MagicMock()
        del bad_engine.initialize  # Remove the method
        valid_config_data["portfolio_engine"] = bad_engine

        config = SimulationConfig(**valid_config_data)
        with pytest.raises(
            ValueError, match="Portfolio engine missing initialize method"
        ):
            config.validate_for_simulation()


class TestConfigEdgeCases:
    """Test edge cases and error conditions."""

    def test_extra_fields_forbidden(self, valid_config_data):
        """Test that extra fields are forbidden."""
        valid_config_data["extra_field"] = "not allowed"

        with pytest.raises(ValueError, match="Extra inputs are not permitted"):
            SimulationConfig(**valid_config_data)

    def test_metadata_field_works(self, valid_config_data):
        """Test that metadata field accepts arbitrary data."""
        valid_config_data["metadata"] = {
            "scenario_name": "Test Scenario",
            "created_by": "Test User",
            "tags": ["retirement", "conservative"],
            "custom_data": {"key": "value"},
        }

        config = SimulationConfig(**valid_config_data)
        assert config.metadata["scenario_name"] == "Test Scenario"
        assert config.metadata["tags"] == ["retirement", "conservative"]

    def test_config_immutability_after_creation(self, valid_config_data):
        """Test that config validates on assignment (validate_assignment=True)."""
        config = SimulationConfig(**valid_config_data)

        # This should trigger validation
        with pytest.raises(ValueError):
            config.num_paths = -100

    def test_provider_type_validation(self, valid_config_data):
        """Test that provider fields require correct types."""
        # Try to pass a string instead of a provider
        bad_provider = "not a provider"
        valid_config_data["returns_provider"] = bad_provider

        # Config creation succeeds due to arbitrary_types_allowed=True
        config = SimulationConfig(**valid_config_data)

        # But validation should fail when we try to validate for simulation
        with pytest.raises(ValueError, match="Returns provider validation failed"):
            config.validate_for_simulation()


class TestConfigIntegration:
    """Integration tests with real-like scenarios."""

    def test_portfolio_only_simulation(self, unit_system, mock_providers):
        """Test configuration for portfolio-only simulation."""
        config = SimulationConfig(
            unit_system=unit_system,
            num_paths=500,
            initial_portfolio_balance=1000000.0,
            target_asset_weights={"asset_0": 0.6, "asset_1": 0.4},
            returns_provider=mock_providers["returns"],
            withdrawal_policy=mock_providers["withdrawal"],
            portfolio_engine=mock_providers["portfolio"],
            rebalancing_strategy=mock_providers["rebalancing"],
        )

        assert config.has_cash_flows() is False
        assert config.has_taxes() is False
        summary = config.create_simulation_summary()
        assert len(summary["providers"]) == 4  # Only required providers

    def test_comprehensive_simulation(self, valid_config_data):
        """Test configuration for comprehensive simulation."""
        config = SimulationConfig(**valid_config_data)

        assert config.has_cash_flows() is True
        assert config.has_taxes() is True
        summary = config.create_simulation_summary()
        assert len(summary["providers"]) == 8  # All providers

    def test_config_with_different_time_horizons(self, mock_providers):
        """Test configuration with different time horizons."""
        # Short-term simulation (5 years)
        short_time_grid = TimeGrid(start_year=2024, end_year=2029, base_year=2024)
        short_inflation = InflationAdjuster(inflation_rate=0.02, base_year=2024)
        short_unit_system = UnitSystem(
            time_grid=short_time_grid,
            inflation_adjuster=short_inflation,
            display_mode="nominal",
        )

        config = SimulationConfig(
            unit_system=short_unit_system,
            num_paths=100,
            initial_portfolio_balance=50000.0,
            target_asset_weights={"asset_0": 0.8, "asset_1": 0.2},
            returns_provider=mock_providers["returns"],
            withdrawal_policy=mock_providers["withdrawal"],
            portfolio_engine=mock_providers["portfolio"],
            rebalancing_strategy=mock_providers["rebalancing"],
        )

        assert config.get_simulation_years() == 6  # 2024-2029 inclusive
        summary = config.create_simulation_summary()
        assert summary["time_period"]["start_year"] == 2024
        assert summary["time_period"]["end_year"] == 2029
        assert summary["display_mode"] == "nominal"
