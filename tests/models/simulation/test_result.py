"""
Tests for simulation result model.

This module tests the SimulationResult Pydantic model including:
- Result data validation
- Helper method functionality
- Statistical analysis methods
- Serialization and export features
"""

import json
import os
import tempfile
from datetime import datetime

import numpy as np
import pytest

from app.models.simulation.result import SimulationResult


@pytest.fixture
def sample_result_data():
    """Create sample simulation result data for testing."""
    years, paths = 10, 100
    num_assets = 2

    # Create sample arrays with realistic data
    np.random.seed(42)  # For reproducible tests

    # Portfolio balances: start at 100k, vary over time
    portfolio_balances = np.zeros((years, paths))
    portfolio_balances[0, :] = 100000  # Initial balance

    # Generate realistic portfolio evolution
    for year in range(1, years):
        returns = np.random.normal(0.07, 0.15, paths)  # 7% mean, 15% volatility
        portfolio_balances[year, :] = portfolio_balances[year - 1, :] * (1 + returns)
        # Subtract some withdrawals
        portfolio_balances[year, :] -= 4000  # $4k withdrawal

    # Asset balances: split between two assets
    asset_balances = np.zeros((num_assets, years, paths))
    for year in range(years):
        asset_balances[0, year, :] = portfolio_balances[year, :] * 0.6  # Stocks
        asset_balances[1, year, :] = portfolio_balances[year, :] * 0.4  # Bonds

    # Other arrays
    withdrawals = np.full((years, paths), 4000.0)  # $4k annual withdrawals
    withdrawals[0, :] = 0  # No withdrawal in first year

    incomes = np.full((years, paths), 50000.0)  # $50k annual income
    incomes[8:, :] = 0  # Stop income after year 8 (retirement)

    expenses = np.full((years, paths), 40000.0)  # $40k annual expenses

    liability_payments = np.full((years, paths), 12000.0)  # $12k mortgage
    liability_payments[6:, :] = 0  # Mortgage paid off after year 6

    taxes = np.full((years, paths), 8000.0)  # $8k annual taxes
    taxes[8:, :] = 0  # No taxes after retirement

    # Portfolio management arrays
    rebalancing_events = np.random.random((years, paths)) < 0.3  # 30% chance
    transaction_costs = np.where(
        rebalancing_events, 100.0, 0.0
    )  # $100 when rebalancing

    portfolio_returns = np.random.normal(0.07, 0.15, (years, paths))
    portfolio_returns[0, :] = 0  # No return in first year

    return {
        "portfolio_balances": portfolio_balances,
        "asset_balances": asset_balances,
        "withdrawals": withdrawals,
        "incomes": incomes,
        "expenses": expenses,
        "liability_payments": liability_payments,
        "taxes": taxes,
        "rebalancing_events": rebalancing_events,
        "transaction_costs": transaction_costs,
        "portfolio_returns": portfolio_returns,
        "asset_names": ["stocks", "bonds"],
        "target_weights": np.array([0.6, 0.4]),
        "simulation_metadata": {
            "scenario_name": "Test Scenario",
            "initial_balance": 100000,
            "withdrawal_strategy": "Fixed $4k",
        },
        "execution_time_seconds": 1.23,
    }


@pytest.fixture
def sample_result(sample_result_data):
    """Create a sample SimulationResult for testing."""
    return SimulationResult(**sample_result_data)


class TestSimulationResultValidation:
    """Test SimulationResult validation."""

    def test_valid_result_creation(self, sample_result_data):
        """Test creating a valid result."""
        result = SimulationResult(**sample_result_data)

        assert result.years == 10
        assert result.paths == 100
        assert result.num_assets == 2
        assert result.shape == (10, 100)
        assert result.asset_names == ["stocks", "bonds"]

    def test_minimal_result_creation(self):
        """Test creating a minimal result with only required fields."""
        years, paths = 5, 10

        result = SimulationResult(
            portfolio_balances=np.random.random((years, paths)),
            asset_balances=np.random.random((2, years, paths)),
            withdrawals=np.random.random((years, paths)),
            rebalancing_events=np.random.random((years, paths)) < 0.5,
            transaction_costs=np.random.random((years, paths)),
            portfolio_returns=np.random.random((years, paths)),
            asset_names=["asset1", "asset2"],
            target_weights=np.array([0.5, 0.5]),
        )

        assert result.incomes is None
        assert result.expenses is None
        assert result.liability_payments is None
        assert result.taxes is None

    def test_invalid_portfolio_balances_shape(self):
        """Test validation of portfolio_balances shape."""
        with pytest.raises(ValueError, match="Array must be 2-dimensional"):
            SimulationResult(
                portfolio_balances=np.random.random((10,)),  # 1D instead of 2D
                asset_balances=np.random.random((2, 10, 5)),
                withdrawals=np.random.random((10, 5)),
                rebalancing_events=np.random.random((10, 5)) < 0.5,
                transaction_costs=np.random.random((10, 5)),
                portfolio_returns=np.random.random((10, 5)),
                asset_names=["asset1", "asset2"],
                target_weights=np.array([0.5, 0.5]),
            )

    def test_invalid_asset_balances_shape(self):
        """Test validation of asset_balances shape."""
        with pytest.raises(ValueError, match="asset_balances must be 3-dimensional"):
            SimulationResult(
                portfolio_balances=np.random.random((10, 5)),
                asset_balances=np.random.random((10, 5)),  # 2D instead of 3D
                withdrawals=np.random.random((10, 5)),
                rebalancing_events=np.random.random((10, 5)) < 0.5,
                transaction_costs=np.random.random((10, 5)),
                portfolio_returns=np.random.random((10, 5)),
                asset_names=["asset1", "asset2"],
                target_weights=np.array([0.5, 0.5]),
            )

    def test_invalid_rebalancing_events_dtype(self):
        """Test validation of rebalancing_events dtype."""
        with pytest.raises(
            ValueError, match="rebalancing_events must be boolean array"
        ):
            SimulationResult(
                portfolio_balances=np.random.random((10, 5)),
                asset_balances=np.random.random((2, 10, 5)),
                withdrawals=np.random.random((10, 5)),
                rebalancing_events=np.random.random((10, 5)),  # Float instead of bool
                transaction_costs=np.random.random((10, 5)),
                portfolio_returns=np.random.random((10, 5)),
                asset_names=["asset1", "asset2"],
                target_weights=np.array([0.5, 0.5]),
            )

    def test_invalid_optional_array_shape(self):
        """Test validation of optional array shapes."""
        with pytest.raises(ValueError, match="Array must be 2-dimensional"):
            SimulationResult(
                portfolio_balances=np.random.random((10, 5)),
                asset_balances=np.random.random((2, 10, 5)),
                withdrawals=np.random.random((10, 5)),
                incomes=np.random.random((10,)),  # 1D instead of 2D
                rebalancing_events=np.random.random((10, 5)) < 0.5,
                transaction_costs=np.random.random((10, 5)),
                portfolio_returns=np.random.random((10, 5)),
                asset_names=["asset1", "asset2"],
                target_weights=np.array([0.5, 0.5]),
            )


class TestResultProperties:
    """Test computed properties of SimulationResult."""

    def test_computed_properties(self, sample_result):
        """Test computed properties."""
        assert sample_result.shape == (10, 100)
        assert sample_result.years == 10
        assert sample_result.paths == 100
        assert sample_result.num_assets == 2

    def test_get_final_balances(self, sample_result):
        """Test getting final balances."""
        final_balances = sample_result.get_final_balances()

        assert final_balances.shape == (100,)  # One value per path
        assert isinstance(final_balances, np.ndarray)
        # Should be the last year's balances
        np.testing.assert_array_equal(
            final_balances, sample_result.portfolio_balances[-1, :]
        )

    def test_get_final_asset_balances(self, sample_result):
        """Test getting final asset balances."""
        final_asset_balances = sample_result.get_final_asset_balances()

        assert final_asset_balances.shape == (2, 100)  # 2 assets × 100 paths
        # Should be the last year's asset balances
        np.testing.assert_array_equal(
            final_asset_balances, sample_result.asset_balances[:, -1, :]
        )


class TestSuccessRateCalculation:
    """Test success rate calculation methods."""

    def test_calculate_success_rate_final_balance(self, sample_result):
        """Test success rate calculation based on final balance."""
        # Test with threshold of 0 (not depleted)
        success_rate = sample_result.calculate_success_rate(threshold=0)
        assert 0 <= success_rate <= 1

        # Test with higher threshold
        success_rate_high = sample_result.calculate_success_rate(threshold=50000)
        assert success_rate_high <= success_rate  # Should be lower or equal

    def test_calculate_success_rate_never_depleted(self, sample_result):
        """Test success rate calculation for never depleted."""
        success_rate = sample_result.calculate_success_rate(
            threshold=0, success_definition="never_depleted"
        )
        assert 0 <= success_rate <= 1

    def test_calculate_success_rate_withdrawal_sustained(self, sample_result):
        """Test success rate calculation for withdrawal sustained."""
        success_rate = sample_result.calculate_success_rate(
            success_definition="withdrawal_sustained"
        )
        assert 0 <= success_rate <= 1

    def test_invalid_success_definition(self, sample_result):
        """Test invalid success definition."""
        with pytest.raises(ValueError, match="Unknown success definition"):
            sample_result.calculate_success_rate(success_definition="invalid")


class TestPercentileCalculation:
    """Test percentile calculation methods."""

    def test_get_percentiles_final_balance(self, sample_result):
        """Test percentile calculation for final balance."""
        percentiles = sample_result.get_percentiles([10, 50, 90], "final_balance")

        assert len(percentiles) == 3
        assert 10 in percentiles
        assert 50 in percentiles
        assert 90 in percentiles

        # 10th percentile should be less than 90th percentile
        assert percentiles[10] <= percentiles[50] <= percentiles[90]

    def test_get_percentiles_different_data_types(self, sample_result):
        """Test percentiles for different data types."""
        data_types = [
            "final_balance",
            "min_balance",
            "max_balance",
            "total_withdrawals",
            "total_returns",
        ]

        for data_type in data_types:
            percentiles = sample_result.get_percentiles([25, 75], data_type)
            assert len(percentiles) == 2
            assert 25 in percentiles
            assert 75 in percentiles

    def test_invalid_data_type(self, sample_result):
        """Test invalid data type for percentiles."""
        with pytest.raises(ValueError, match="Unknown data type"):
            sample_result.get_percentiles([50], "invalid_type")


class TestStatisticsMethods:
    """Test various statistics calculation methods."""

    def test_get_balance_statistics(self, sample_result):
        """Test balance statistics calculation."""
        stats = sample_result.get_balance_statistics()

        expected_keys = [
            "final_balance_mean",
            "final_balance_median",
            "final_balance_std",
            "min_balance_mean",
            "min_balance_median",
            "max_balance_mean",
            "max_balance_median",
            "depletion_rate",
        ]

        for key in expected_keys:
            assert key in stats
            assert isinstance(stats[key], (int, float))

    def test_get_withdrawal_statistics(self, sample_result):
        """Test withdrawal statistics calculation."""
        stats = sample_result.get_withdrawal_statistics()

        expected_keys = [
            "total_withdrawal_mean",
            "total_withdrawal_median",
            "annual_withdrawal_mean",
            "annual_withdrawal_std",
            "max_annual_withdrawal",
            "min_annual_withdrawal",
        ]

        for key in expected_keys:
            assert key in stats
            assert isinstance(stats[key], (int, float))

    def test_get_rebalancing_statistics(self, sample_result):
        """Test rebalancing statistics calculation."""
        stats = sample_result.get_rebalancing_statistics()

        expected_keys = [
            "total_rebalancing_events",
            "rebalancing_frequency",
            "total_transaction_costs",
            "avg_transaction_cost_per_event",
            "transaction_cost_as_pct_of_portfolio",
        ]

        for key in expected_keys:
            assert key in stats
            assert isinstance(stats[key], (int, float))

    def test_get_cash_flow_statistics(self, sample_result):
        """Test cash flow statistics calculation."""
        stats = sample_result.get_cash_flow_statistics()

        # Should include income, expenses, liability, and tax stats
        assert "total_income_mean" in stats
        assert "total_expenses_mean" in stats
        assert "total_liability_payments" in stats
        assert "total_taxes_mean" in stats

    def test_get_cash_flow_statistics_minimal(self):
        """Test cash flow statistics with minimal data (no optional arrays)."""
        years, paths = 5, 10

        result = SimulationResult(
            portfolio_balances=np.random.random((years, paths)),
            asset_balances=np.random.random((2, years, paths)),
            withdrawals=np.random.random((years, paths)),
            rebalancing_events=np.random.random((years, paths)) < 0.5,
            transaction_costs=np.random.random((years, paths)),
            portfolio_returns=np.random.random((years, paths)),
            asset_names=["asset1", "asset2"],
            target_weights=np.array([0.5, 0.5]),
        )

        stats = result.get_cash_flow_statistics()
        # Should be empty since no cash flow arrays provided
        assert isinstance(stats, dict)

    def test_get_return_statistics(self, sample_result):
        """Test return statistics calculation."""
        stats = sample_result.get_return_statistics()

        expected_keys = [
            "annualized_return_mean",
            "annualized_return_median",
            "annualized_return_std",
            "annual_return_mean",
            "annual_return_std",
            "best_annual_return",
            "worst_annual_return",
        ]

        for key in expected_keys:
            assert key in stats
            assert isinstance(stats[key], (int, float))


class TestSummaryReport:
    """Test summary report generation."""

    def test_create_summary_report(self, sample_result):
        """Test comprehensive summary report creation."""
        report = sample_result.create_summary_report()

        expected_sections = [
            "simulation_info",
            "success_metrics",
            "balance_statistics",
            "withdrawal_statistics",
            "return_statistics",
            "rebalancing_statistics",
            "cash_flow_statistics",
            "percentiles",
            "metadata",
        ]

        for section in expected_sections:
            assert section in report

        # Check simulation info
        sim_info = report["simulation_info"]
        assert sim_info["years"] == 10
        assert sim_info["paths"] == 100
        assert sim_info["assets"] == ["stocks", "bonds"]

        # Check success metrics
        success_metrics = report["success_metrics"]
        assert "success_rate_not_depleted" in success_metrics
        assert "success_rate_positive_balance" in success_metrics

        # Check percentiles
        percentiles = report["percentiles"]
        assert "final_balance" in percentiles
        assert "min_balance" in percentiles


class TestSerialization:
    """Test serialization and export functionality."""

    def test_to_dict_without_arrays(self, sample_result):
        """Test conversion to dictionary without arrays."""
        result_dict = sample_result.to_dict(include_arrays=False)

        assert "simulation_info" in result_dict
        assert "summary_statistics" in result_dict
        assert "arrays" not in result_dict

    def test_to_dict_with_arrays(self, sample_result):
        """Test conversion to dictionary with arrays."""
        result_dict = sample_result.to_dict(include_arrays=True)

        assert "simulation_info" in result_dict
        assert "summary_statistics" in result_dict
        assert "arrays" in result_dict

        arrays = result_dict["arrays"]
        assert "portfolio_balances" in arrays
        assert "asset_balances" in arrays
        assert "withdrawals" in arrays
        assert "incomes" in arrays  # Optional array present
        assert "expenses" in arrays  # Optional array present

    def test_to_json(self, sample_result):
        """Test JSON serialization."""
        json_str = sample_result.to_json(include_arrays=False)

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert "simulation_info" in parsed

    def test_save_to_file(self, sample_result):
        """Test saving to file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            filepath = f.name

        try:
            sample_result.save_to_file(filepath, include_arrays=True)

            # Verify file was created and contains valid JSON
            assert os.path.exists(filepath)

            with open(filepath, "r") as f:
                data = json.load(f)

            assert isinstance(data, dict)
            assert "simulation_info" in data
            assert "arrays" in data

        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)


class TestPathDataExtraction:
    """Test path-specific data extraction."""

    def test_get_path_data(self, sample_result):
        """Test extracting data for a specific path."""
        path_data = sample_result.get_path_data(0)

        expected_keys = [
            "portfolio_balances",
            "asset_balances",
            "withdrawals",
            "rebalancing_events",
            "transaction_costs",
            "portfolio_returns",
            "incomes",
            "expenses",
            "liability_payments",
            "taxes",
        ]

        for key in expected_keys:
            assert key in path_data
            assert isinstance(path_data[key], np.ndarray)

        # Check shapes
        assert path_data["portfolio_balances"].shape == (10,)  # years
        assert path_data["asset_balances"].shape == (2, 10)  # assets × years
        assert path_data["withdrawals"].shape == (10,)  # years

    def test_get_path_data_invalid_index(self, sample_result):
        """Test getting path data with invalid index."""
        with pytest.raises(ValueError, match="Path index 200 >= number of paths 100"):
            sample_result.get_path_data(200)


class TestResultComparison:
    """Test result comparison functionality."""

    def test_compare_with_same_result(self, sample_result):
        """Test comparing result with itself."""
        comparison = sample_result.compare_with(sample_result)

        assert "success_rate_difference" in comparison
        assert "final_balance_difference" in comparison
        assert "withdrawal_difference" in comparison

        # Comparing with itself should give zero differences
        assert comparison["success_rate_difference"] == 0
        assert comparison["final_balance_difference"]["mean"] == 0
        assert comparison["withdrawal_difference"]["mean"] == 0

    def test_compare_with_different_shapes(self, sample_result):
        """Test comparing results with different shapes."""
        # Create a different sized result
        different_result = SimulationResult(
            portfolio_balances=np.random.random((5, 50)),  # Different shape
            asset_balances=np.random.random((2, 5, 50)),
            withdrawals=np.random.random((5, 50)),
            rebalancing_events=np.random.random((5, 50)) < 0.5,
            transaction_costs=np.random.random((5, 50)),
            portfolio_returns=np.random.random((5, 50)),
            asset_names=["asset1", "asset2"],
            target_weights=np.array([0.5, 0.5]),
        )

        with pytest.raises(
            ValueError, match="Cannot compare results with different shapes"
        ):
            sample_result.compare_with(different_result)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_created_at_default(self, sample_result_data):
        """Test that created_at defaults to current time."""
        result = SimulationResult(**sample_result_data)

        assert isinstance(result.created_at, datetime)
        # Should be recent (within last minute)
        time_diff = datetime.now() - result.created_at
        assert time_diff.total_seconds() < 60

    def test_extra_fields_forbidden(self, sample_result_data):
        """Test that extra fields are forbidden."""
        sample_result_data["extra_field"] = "not allowed"

        with pytest.raises(ValueError, match="Extra inputs are not permitted"):
            SimulationResult(**sample_result_data)

    def test_metadata_field_works(self, sample_result_data):
        """Test that metadata field accepts arbitrary data."""
        sample_result_data["simulation_metadata"] = {
            "custom_field": "custom_value",
            "nested": {"key": "value"},
            "list": [1, 2, 3],
        }

        result = SimulationResult(**sample_result_data)
        assert result.simulation_metadata["custom_field"] == "custom_value"
        assert result.simulation_metadata["nested"]["key"] == "value"

    def test_detailed_ledger_optional(self, sample_result_data):
        """Test that detailed_ledger is optional."""
        result = SimulationResult(**sample_result_data)
        assert result.detailed_ledger is None

        # Test with detailed ledger
        sample_result_data["detailed_ledger"] = [
            {"year": 0, "action": "initial", "amount": 100000},
            {"year": 1, "action": "withdrawal", "amount": -4000},
        ]

        result_with_ledger = SimulationResult(**sample_result_data)
        assert len(result_with_ledger.detailed_ledger) == 2


class TestIntegrationScenarios:
    """Integration tests with realistic scenarios."""

    def test_portfolio_only_result(self):
        """Test result for portfolio-only simulation (minimal data)."""
        years, paths = 20, 1000

        result = SimulationResult(
            portfolio_balances=np.random.uniform(50000, 200000, (years, paths)),
            asset_balances=np.random.uniform(10000, 100000, (3, years, paths)),
            withdrawals=np.random.uniform(3000, 5000, (years, paths)),
            rebalancing_events=np.random.random((years, paths)) < 0.2,
            transaction_costs=np.random.uniform(0, 200, (years, paths)),
            portfolio_returns=np.random.normal(0.07, 0.15, (years, paths)),
            asset_names=["stocks", "bonds", "cash"],
            target_weights=np.array([0.6, 0.3, 0.1]),
        )

        # Should work without cash flow data
        stats = result.get_cash_flow_statistics()
        assert isinstance(stats, dict)

        # Should still calculate success rates
        success_rate = result.calculate_success_rate()
        assert 0 <= success_rate <= 1

    def test_comprehensive_result_analysis(self, sample_result):
        """Test comprehensive analysis of a full result."""
        # Generate comprehensive summary
        summary = sample_result.create_summary_report()

        # Verify all sections are present and reasonable
        assert summary["simulation_info"]["years"] == 10
        assert summary["simulation_info"]["paths"] == 100

        # Success metrics should be reasonable
        success_metrics = summary["success_metrics"]
        assert 0 <= success_metrics["success_rate_not_depleted"] <= 1

        # Balance statistics should be positive
        balance_stats = summary["balance_statistics"]
        assert balance_stats["final_balance_mean"] > 0

        # Withdrawal statistics should match our test data
        withdrawal_stats = summary["withdrawal_statistics"]
        assert withdrawal_stats["annual_withdrawal_mean"] > 0

        # Should have cash flow statistics
        cash_flow_stats = summary["cash_flow_statistics"]
        assert "total_income_mean" in cash_flow_stats
