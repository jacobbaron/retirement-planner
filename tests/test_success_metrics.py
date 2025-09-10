"""
Tests for success metrics module.
"""

import numpy as np
import pytest

from app.models.success_metrics import (
    SuccessMetricsCalculator,
    SuccessMetricsConfig,
    SuccessMetricsResult,
    compare_success_metrics,
    generate_success_report,
    validate_success_metrics,
)
from app.models.withdrawal_rules import WithdrawalResult


class TestSuccessMetricsConfig:
    """Test SuccessMetricsConfig validation."""

    def test_valid_config(self):
        """Test valid configuration."""
        config = SuccessMetricsConfig(
            confidence_levels=[0.05, 0.25, 0.50, 0.75, 0.95],
            failure_threshold=0.0,
            success_threshold=0.0,
        )

        assert config.confidence_levels == [0.05, 0.25, 0.50, 0.75, 0.95]
        assert config.failure_threshold == 0.0
        assert config.success_threshold == 0.0

    def test_default_config(self):
        """Test default configuration."""
        config = SuccessMetricsConfig()

        assert config.confidence_levels == [0.05, 0.25, 0.50, 0.75, 0.95]
        assert config.failure_threshold == 0.0
        assert config.success_threshold == 0.0

    def test_invalid_failure_threshold(self):
        """Test invalid failure threshold."""
        with pytest.raises(Exception):  # Pydantic validation error
            SuccessMetricsConfig(failure_threshold=-0.1)

    def test_invalid_success_threshold(self):
        """Test invalid success threshold."""
        with pytest.raises(Exception):  # Pydantic validation error
            SuccessMetricsConfig(success_threshold=-0.1)


class TestSuccessMetricsCalculator:
    """Test SuccessMetricsCalculator functionality."""

    def test_initialization(self):
        """Test calculator initialization."""
        config = SuccessMetricsConfig()
        calculator = SuccessMetricsCalculator(config)

        assert calculator.config == config

    def test_initialization_default_config(self):
        """Test calculator initialization with default config."""
        calculator = SuccessMetricsCalculator()

        assert calculator.config is not None
        assert calculator.config.confidence_levels == [0.05, 0.25, 0.50, 0.75, 0.95]

    def test_calculate_metrics_basic(self):
        """Test basic metrics calculation."""
        # Create mock withdrawal result
        years = 5
        num_paths = 100

        # Create mock data
        portfolio_balances = np.random.normal(100000, 20000, (years, num_paths))
        portfolio_balances[0, :] = 100000  # Set initial balance
        withdrawals = np.random.normal(4000, 500, (years, num_paths))
        failures = np.zeros((years, num_paths), dtype=bool)
        first_failure_years = np.full(num_paths, -1, dtype=np.int32)
        success_rate = 1.0

        withdrawal_result = WithdrawalResult(
            withdrawals=withdrawals,
            portfolio_balances=portfolio_balances,
            failures=failures,
            first_failure_years=first_failure_years,
            success_rate=success_rate,
        )

        # Calculate metrics
        calculator = SuccessMetricsCalculator()
        result = calculator.calculate_metrics(withdrawal_result)

        # Verify result structure
        assert isinstance(result, SuccessMetricsResult)
        assert result.success_rate == 1.0
        assert "p5" in result.terminal_wealth_percentiles
        assert "p50" in result.terminal_wealth_percentiles
        assert "p95" in result.terminal_wealth_percentiles
        assert "mean" in result.first_failure_year_stats
        assert "mean" in result.balance_statistics
        assert "total_withdrawals_mean" in result.withdrawal_statistics
        assert "max_drawdown" in result.risk_metrics

    def test_calculate_success_rate(self):
        """Test success rate calculation."""
        calculator = SuccessMetricsCalculator()

        # Create mock result with 80% success rate
        years = 5
        num_paths = 100
        portfolio_balances = np.ones((years, num_paths)) * 100000
        withdrawals = np.ones((years, num_paths)) * 4000
        failures = np.zeros((years, num_paths), dtype=bool)
        first_failure_years = np.full(num_paths, -1, dtype=np.int32)
        first_failure_years[20:] = np.random.randint(1, 5, 80)  # 80 failures
        success_rate = 0.2

        withdrawal_result = WithdrawalResult(
            withdrawals=withdrawals,
            portfolio_balances=portfolio_balances,
            failures=failures,
            first_failure_years=first_failure_years,
            success_rate=success_rate,
        )

        result = calculator.calculate_metrics(withdrawal_result)
        assert result.success_rate == 0.2

    def test_calculate_terminal_wealth_percentiles(self):
        """Test terminal wealth percentile calculation."""
        calculator = SuccessMetricsCalculator()

        # Create mock result with known final balances
        years = 5
        num_paths = 100
        portfolio_balances = np.ones((years, num_paths)) * 100000
        portfolio_balances[-1, :] = np.linspace(
            50000, 150000, num_paths
        )  # Known distribution
        withdrawals = np.ones((years, num_paths)) * 4000
        failures = np.zeros((years, num_paths), dtype=bool)
        first_failure_years = np.full(num_paths, -1, dtype=np.int32)
        success_rate = 1.0

        withdrawal_result = WithdrawalResult(
            withdrawals=withdrawals,
            portfolio_balances=portfolio_balances,
            failures=failures,
            first_failure_years=first_failure_years,
            success_rate=success_rate,
        )

        result = calculator.calculate_metrics(withdrawal_result)

        # Verify percentiles are calculated correctly (allowing for small numerical differences)
        assert np.isclose(result.terminal_wealth_percentiles["p5"], 55000.0, atol=1000)
        assert np.isclose(
            result.terminal_wealth_percentiles["p50"], 100000.0, atol=1000
        )
        assert np.isclose(
            result.terminal_wealth_percentiles["p95"], 145000.0, atol=1000
        )

    def test_calculate_first_failure_year_stats(self):
        """Test first failure year statistics calculation."""
        calculator = SuccessMetricsCalculator()

        # Create mock result with known failure years
        years = 5
        num_paths = 100
        portfolio_balances = np.ones((years, num_paths)) * 100000
        withdrawals = np.ones((years, num_paths)) * 4000
        failures = np.zeros((years, num_paths), dtype=bool)
        first_failure_years = np.full(num_paths, -1, dtype=np.int32)
        first_failure_years[50:] = [1, 2, 3, 4, 5] * 10  # 50 failures with known years
        success_rate = 0.5

        withdrawal_result = WithdrawalResult(
            withdrawals=withdrawals,
            portfolio_balances=portfolio_balances,
            failures=failures,
            first_failure_years=first_failure_years,
            success_rate=success_rate,
        )

        result = calculator.calculate_metrics(withdrawal_result)

        # Verify failure year statistics
        assert result.first_failure_year_stats["count"] == 50.0
        assert result.first_failure_year_stats["mean"] == 3.0  # (1+2+3+4+5)/5
        assert result.first_failure_year_stats["min"] == 1.0
        assert result.first_failure_year_stats["max"] == 5.0

    def test_calculate_balance_statistics(self):
        """Test portfolio balance statistics over time."""
        calculator = SuccessMetricsCalculator()

        # Create mock result with known balance progression
        years = 3
        num_paths = 3  # Use 3 paths to match the array structure
        portfolio_balances = np.array(
            [
                [100000, 100000, 100000],  # Year 0
                [105000, 105000, 105000],  # Year 1
                [110000, 110000, 110000],  # Year 2
            ]
        )  # Shape: (years, num_paths)
        withdrawals = np.ones((years, num_paths)) * 4000
        failures = np.zeros((years, num_paths), dtype=bool)
        first_failure_years = np.full(num_paths, -1, dtype=np.int32)
        success_rate = 1.0

        withdrawal_result = WithdrawalResult(
            withdrawals=withdrawals,
            portfolio_balances=portfolio_balances,
            failures=failures,
            first_failure_years=first_failure_years,
            success_rate=success_rate,
        )

        result = calculator.calculate_metrics(withdrawal_result)

        # Verify balance statistics
        assert np.allclose(result.balance_statistics["mean"], [100000, 105000, 110000])
        assert np.allclose(result.balance_statistics["std"], [0, 0, 0])  # No variance
        assert np.allclose(result.balance_statistics["p50"], [100000, 105000, 110000])

    def test_calculate_withdrawal_statistics(self):
        """Test withdrawal statistics calculation."""
        calculator = SuccessMetricsCalculator()

        # Create mock result with known withdrawals
        years = 3
        num_paths = 100
        portfolio_balances = np.ones((years, num_paths)) * 100000
        withdrawals = np.array(
            [
                [4000, 4000, 4000],  # Year 0
                [4200, 4200, 4200],  # Year 1
                [4400, 4400, 4400],  # Year 2
            ]
        ).T
        failures = np.zeros((years, num_paths), dtype=bool)
        first_failure_years = np.full(num_paths, -1, dtype=np.int32)
        success_rate = 1.0

        withdrawal_result = WithdrawalResult(
            withdrawals=withdrawals,
            portfolio_balances=portfolio_balances,
            failures=failures,
            first_failure_years=first_failure_years,
            success_rate=success_rate,
        )

        result = calculator.calculate_metrics(withdrawal_result)

        # Verify withdrawal statistics
        assert (
            result.withdrawal_statistics["total_withdrawals_mean"] == 12600.0
        )  # 4000+4200+4400
        assert (
            result.withdrawal_statistics["avg_annual_withdrawal_mean"] == 4200.0
        )  # (4000+4200+4400)/3

    def test_calculate_risk_metrics(self):
        """Test risk metrics calculation."""
        calculator = SuccessMetricsCalculator()

        # Create mock result with known portfolio evolution
        years = 3
        num_paths = 3  # Use 3 paths to match the array structure
        portfolio_balances = np.array(
            [
                [100000, 100000, 100000],  # Year 0
                [95000, 95000, 95000],  # Year 1 (5% drawdown)
                [100000, 100000, 100000],  # Year 2 (recovery)
            ]
        )  # Shape: (years, num_paths)
        withdrawals = np.ones((years, num_paths)) * 4000
        failures = np.zeros((years, num_paths), dtype=bool)
        first_failure_years = np.full(num_paths, -1, dtype=np.int32)
        success_rate = 1.0

        withdrawal_result = WithdrawalResult(
            withdrawals=withdrawals,
            portfolio_balances=portfolio_balances,
            failures=failures,
            first_failure_years=first_failure_years,
            success_rate=success_rate,
        )

        result = calculator.calculate_metrics(withdrawal_result)

        # Verify risk metrics
        assert result.risk_metrics["max_drawdown"] == 0.05  # 5% drawdown
        assert result.risk_metrics["failure_rate"] == 0.0  # No failures

    def test_calculate_max_drawdown(self):
        """Test maximum drawdown calculation."""
        calculator = SuccessMetricsCalculator()

        # Create portfolio balances with known drawdown pattern
        portfolio_balances = np.array(
            [
                [100000, 100000, 100000],  # Year 0 (peak)
                [95000, 90000, 85000],  # Year 1 (drawdown)
                [100000, 95000, 90000],  # Year 2 (partial recovery)
            ]
        )  # Shape: (years, num_paths)

        max_drawdown = calculator._calculate_max_drawdown(portfolio_balances)

        # Path 0: 5% drawdown, Path 1: 10% drawdown, Path 2: 15% drawdown
        # Maximum drawdown across all paths is 15%
        assert max_drawdown == 0.15

    def test_calculate_metrics_with_portfolio_returns(self):
        """Test metrics calculation with portfolio returns."""
        calculator = SuccessMetricsCalculator()

        # Create mock data
        years = 3
        num_paths = 100
        portfolio_balances = np.random.normal(100000, 20000, (years, num_paths))
        portfolio_balances[0, :] = 100000
        withdrawals = np.ones((years, num_paths)) * 4000
        failures = np.zeros((years, num_paths), dtype=bool)
        first_failure_years = np.full(num_paths, -1, dtype=np.int32)
        success_rate = 1.0

        withdrawal_result = WithdrawalResult(
            withdrawals=withdrawals,
            portfolio_balances=portfolio_balances,
            failures=failures,
            first_failure_years=first_failure_years,
            success_rate=success_rate,
        )

        # Create mock portfolio returns
        portfolio_returns = np.random.normal(0.05, 0.15, (years - 1, num_paths))

        result = calculator.calculate_metrics(withdrawal_result, portfolio_returns)

        # Verify portfolio volatility is calculated
        assert "portfolio_volatility" in result.risk_metrics
        assert result.risk_metrics["portfolio_volatility"] > 0


class TestSuccessMetricsComparison:
    """Test success metrics comparison functionality."""

    def test_compare_success_metrics(self):
        """Test comparison of success metrics across strategies."""
        # Create mock results for two strategies
        years = 3
        num_paths = 100

        # Strategy 1: High success rate
        portfolio_balances_1 = np.ones((years, num_paths)) * 100000
        withdrawals_1 = np.ones((years, num_paths)) * 4000
        failures_1 = np.zeros((years, num_paths), dtype=bool)
        first_failure_years_1 = np.full(num_paths, -1, dtype=np.int32)
        success_rate_1 = 0.9

        withdrawal_result_1 = WithdrawalResult(
            withdrawals=withdrawals_1,
            portfolio_balances=portfolio_balances_1,
            failures=failures_1,
            first_failure_years=first_failure_years_1,
            success_rate=success_rate_1,
        )

        calculator = SuccessMetricsCalculator()
        result_1 = calculator.calculate_metrics(withdrawal_result_1)

        # Strategy 2: Lower success rate
        portfolio_balances_2 = np.ones((years, num_paths)) * 100000
        withdrawals_2 = np.ones((years, num_paths)) * 5000
        failures_2 = np.zeros((years, num_paths), dtype=bool)
        first_failure_years_2 = np.full(num_paths, -1, dtype=np.int32)
        first_failure_years_2[50:] = 2  # 50% failures at year 2
        success_rate_2 = 0.5

        withdrawal_result_2 = WithdrawalResult(
            withdrawals=withdrawals_2,
            portfolio_balances=portfolio_balances_2,
            failures=failures_2,
            first_failure_years=first_failure_years_2,
            success_rate=success_rate_2,
        )

        result_2 = calculator.calculate_metrics(withdrawal_result_2)

        # Compare strategies
        comparison = compare_success_metrics(
            [
                ("Strategy 1", result_1),
                ("Strategy 2", result_2),
            ]
        )

        # Verify comparison structure
        assert "Strategy 1" in comparison
        assert "Strategy 2" in comparison
        assert comparison["Strategy 1"]["success_rate"] == 0.9
        assert comparison["Strategy 2"]["success_rate"] == 0.5


class TestSuccessReport:
    """Test success report generation."""

    def test_generate_success_report(self):
        """Test success report generation."""
        # Create mock result
        years = 3
        num_paths = 100
        portfolio_balances = np.ones((years, num_paths)) * 100000
        withdrawals = np.ones((years, num_paths)) * 4000
        failures = np.zeros((years, num_paths), dtype=bool)
        first_failure_years = np.full(num_paths, -1, dtype=np.int32)
        success_rate = 0.8

        withdrawal_result = WithdrawalResult(
            withdrawals=withdrawals,
            portfolio_balances=portfolio_balances,
            failures=failures,
            first_failure_years=first_failure_years,
            success_rate=success_rate,
        )

        calculator = SuccessMetricsCalculator()
        result = calculator.calculate_metrics(withdrawal_result)

        # Generate report
        report = generate_success_report(result, "Test Strategy")

        # Verify report contains expected sections
        assert "Test Strategy Success Report" in report
        assert "Success Rate: 80.0%" in report
        assert "Terminal Wealth Percentiles:" in report
        assert "First Failure Year Statistics:" in report
        assert "Risk Metrics:" in report
        assert "Withdrawal Statistics:" in report


class TestSuccessMetricsValidation:
    """Test success metrics validation."""

    def test_validate_success_metrics_success(self):
        """Test successful validation."""
        # Create mock result
        years = 3
        num_paths = 100
        portfolio_balances = np.ones((years, num_paths)) * 100000
        withdrawals = np.ones((years, num_paths)) * 4000
        failures = np.zeros((years, num_paths), dtype=bool)
        first_failure_years = np.full(num_paths, -1, dtype=np.int32)
        success_rate = 1.0

        withdrawal_result = WithdrawalResult(
            withdrawals=withdrawals,
            portfolio_balances=portfolio_balances,
            failures=failures,
            first_failure_years=first_failure_years,
            success_rate=success_rate,
        )

        calculator = SuccessMetricsCalculator()
        result = calculator.calculate_metrics(withdrawal_result)

        # Validate with expected properties
        expected_properties = {"success_rate": 1.0}
        is_valid, message = validate_success_metrics(result, expected_properties)

        assert is_valid
        assert "within tolerance" in message

    def test_validate_success_metrics_failure(self):
        """Test validation failure."""
        # Create mock result
        years = 3
        num_paths = 100
        portfolio_balances = np.ones((years, num_paths)) * 100000
        withdrawals = np.ones((years, num_paths)) * 4000
        failures = np.zeros((years, num_paths), dtype=bool)
        first_failure_years = np.full(num_paths, -1, dtype=np.int32)
        success_rate = 1.0

        withdrawal_result = WithdrawalResult(
            withdrawals=withdrawals,
            portfolio_balances=portfolio_balances,
            failures=failures,
            first_failure_years=first_failure_years,
            success_rate=success_rate,
        )

        calculator = SuccessMetricsCalculator()
        result = calculator.calculate_metrics(withdrawal_result)

        # Validate with incorrect expected properties
        expected_properties = {"success_rate": 0.5}  # Should be 1.0
        is_valid, message = validate_success_metrics(result, expected_properties)

        assert not is_valid
        assert "expected" in message and "got" in message


class TestIntegrationWithWithdrawalRules:
    """Test integration with withdrawal rules."""

    def test_integration_with_withdrawal_rules(self):
        """Test integration with withdrawal rules module."""
        from app.models.portfolio_evolution import (
            PortfolioEvolution,
            PortfolioEvolutionConfig,
        )
        from app.models.random_returns import AssetClass
        from app.models.withdrawal_rules import (
            FixedRealWithdrawalRule,
            WithdrawalRuleConfig,
        )

        # Create asset classes
        asset_classes = [
            AssetClass(
                name="Stocks", expected_return=0.07, volatility=0.18, weight=0.6
            ),
            AssetClass(name="Bonds", expected_return=0.03, volatility=0.06, weight=0.4),
        ]

        # Create portfolio evolution config
        portfolio_config = PortfolioEvolutionConfig(
            initial_balance=100000,
            asset_classes=asset_classes,
            years=10,
            num_paths=100,
        )

        # Run portfolio evolution
        evolution = PortfolioEvolution(portfolio_config)
        portfolio_result = evolution.simulate()

        # Create withdrawal rule config
        withdrawal_config = WithdrawalRuleConfig(
            initial_balance=100000,
            years=10,
            num_paths=100,
        )

        # Test withdrawal rule
        withdrawal_rule = FixedRealWithdrawalRule(
            withdrawal_config, initial_withdrawal_rate=0.04
        )

        # Apply withdrawal rule
        withdrawal_result = withdrawal_rule.apply_withdrawals(portfolio_result)

        # Calculate success metrics
        calculator = SuccessMetricsCalculator()
        metrics_result = calculator.calculate_metrics(
            withdrawal_result, portfolio_result.portfolio_returns
        )

        # Verify integration works
        assert isinstance(metrics_result, SuccessMetricsResult)
        assert metrics_result.success_rate >= 0.0
        assert metrics_result.success_rate <= 1.0
        assert "p5" in metrics_result.terminal_wealth_percentiles
        assert "p50" in metrics_result.terminal_wealth_percentiles
        assert "p95" in metrics_result.terminal_wealth_percentiles
        assert "max_drawdown" in metrics_result.risk_metrics
