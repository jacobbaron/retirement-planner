"""
Tests for withdrawal rules module.
"""

import numpy as np
import pytest

from app.models.portfolio_evolution import PortfolioEvolutionResult
from app.models.random_returns import AssetClass
from app.models.withdrawal_rules import (
    FixedPercentageWithdrawalRule,
    FixedRealWithdrawalRule,
    VPWWithdrawalRule,
    WithdrawalRuleConfig,
    calculate_withdrawal_statistics,
    compare_withdrawal_strategies,
    validate_withdrawal_rule,
)


class TestWithdrawalRuleConfig:
    """Test WithdrawalRuleConfig validation."""

    def test_valid_config(self):
        """Test valid configuration."""
        config = WithdrawalRuleConfig(
            initial_balance=100000,
            years=30,
            num_paths=1000,
            inflation_rate=0.03,
        )

        assert config.initial_balance == 100000
        assert config.years == 30
        assert config.num_paths == 1000
        assert config.inflation_rate == 0.03

    def test_invalid_initial_balance(self):
        """Test invalid initial balance."""
        with pytest.raises(Exception):  # Pydantic validation error
            WithdrawalRuleConfig(
                initial_balance=0,
                years=30,
                num_paths=1000,
            )

    def test_invalid_years(self):
        """Test invalid number of years."""
        with pytest.raises(Exception):  # Pydantic validation error
            WithdrawalRuleConfig(
                initial_balance=100000,
                years=0,
                num_paths=1000,
            )

    def test_invalid_num_paths(self):
        """Test invalid number of paths."""
        with pytest.raises(Exception):  # Pydantic validation error
            WithdrawalRuleConfig(
                initial_balance=100000,
                years=30,
                num_paths=0,
            )

    def test_invalid_inflation_rate(self):
        """Test invalid inflation rate."""
        with pytest.raises(Exception):  # Pydantic validation error
            WithdrawalRuleConfig(
                initial_balance=100000,
                years=30,
                num_paths=1000,
                inflation_rate=-0.01,
            )


class TestFixedRealWithdrawalRule:
    """Test FixedRealWithdrawalRule (4% rule)."""

    def test_initial_withdrawal_calculation(self):
        """Test initial withdrawal calculation."""
        config = WithdrawalRuleConfig(
            initial_balance=100000,
            years=30,
            num_paths=1,
        )
        rule = FixedRealWithdrawalRule(config, initial_withdrawal_rate=0.04)

        # First year withdrawal should be 4% of initial balance
        withdrawal = rule.calculate_withdrawal(0, 100000, 0)
        assert withdrawal == 4000.0

    def test_inflation_adjustment(self):
        """Test inflation adjustment over time."""
        config = WithdrawalRuleConfig(
            initial_balance=100000,
            years=30,
            num_paths=1,
            inflation_rate=0.03,
        )
        rule = FixedRealWithdrawalRule(config, initial_withdrawal_rate=0.04)

        # Year 0: 4% of 100,000 = 4,000
        withdrawal_0 = rule.calculate_withdrawal(0, 100000, 0)
        assert withdrawal_0 == 4000.0

        # Year 1: 4,000 * 1.03 = 4,120
        withdrawal_1 = rule.calculate_withdrawal(1, 100000, 0)
        assert np.isclose(withdrawal_1, 4120.0)

        # Year 2: 4,000 * 1.03^2 = 4,243.6
        withdrawal_2 = rule.calculate_withdrawal(2, 100000, 0)
        assert np.isclose(withdrawal_2, 4243.6)

    def test_deterministic_no_volatility(self):
        """Test deterministic case with no volatility (toy case)."""
        # Create a simple portfolio with fixed returns
        config = WithdrawalRuleConfig(
            initial_balance=100000,
            years=5,
            num_paths=1,
            inflation_rate=0.03,
        )
        rule = FixedRealWithdrawalRule(config, initial_withdrawal_rate=0.04)

        # Create mock portfolio result with fixed 5% returns
        portfolio_balances = np.zeros((5, 1))
        portfolio_balances[0, 0] = 100000  # Set initial balance
        portfolio_returns = np.full((4, 1), 0.05)  # 5% return each year
        asset_balances = np.zeros((1, 5, 1))
        asset_balances[0, 0, 0] = 100000  # Set initial asset balance
        rebalancing_events = np.zeros((5, 1), dtype=bool)
        transaction_costs = np.zeros((5, 1))
        target_weights = np.array([1.0])
        asset_names = ["Portfolio"]

        portfolio_result = PortfolioEvolutionResult(
            portfolio_balances=portfolio_balances,
            asset_balances=asset_balances,
            rebalancing_events=rebalancing_events,
            transaction_costs=transaction_costs,
            portfolio_returns=portfolio_returns,
            target_weights=target_weights,
            asset_names=asset_names,
        )

        # Apply withdrawals
        result = rule.apply_withdrawals(portfolio_result)

        # Verify deterministic results
        # Year 0: Start with 100,000, withdraw 4,000, balance = 96,000
        assert np.isclose(result.portfolio_balances[0, 0], 96000.0)
        assert np.isclose(result.withdrawals[0, 0], 4000.0)

        # Year 1: 96,000 * 1.05 = 100,800, withdraw 4,120, balance = 96,680
        assert np.isclose(result.portfolio_balances[1, 0], 96680.0)
        assert np.isclose(result.withdrawals[1, 0], 4120.0)

        # Year 2: 96,680 * 1.05 = 101,514, withdraw 4,243.6, balance = 97,270.4
        assert np.isclose(result.portfolio_balances[2, 0], 97270.4)
        assert np.isclose(result.withdrawals[2, 0], 4243.6)

        # Should not fail in this scenario
        assert result.success_rate == 1.0
        assert result.first_failure_years[0] == -1


class TestFixedPercentageWithdrawalRule:
    """Test FixedPercentageWithdrawalRule."""

    def test_withdrawal_calculation(self):
        """Test withdrawal calculation as percentage of balance."""
        config = WithdrawalRuleConfig(
            initial_balance=100000,
            years=30,
            num_paths=1,
        )
        rule = FixedPercentageWithdrawalRule(config, withdrawal_rate=0.05)

        # Withdrawal should be 5% of current balance
        withdrawal = rule.calculate_withdrawal(0, 100000, 0)
        assert withdrawal == 5000.0

        withdrawal = rule.calculate_withdrawal(1, 50000, 0)
        assert withdrawal == 2500.0

    def test_deterministic_no_volatility(self):
        """Test deterministic case with no volatility (toy case)."""
        config = WithdrawalRuleConfig(
            initial_balance=100000,
            years=5,
            num_paths=1,
        )
        rule = FixedPercentageWithdrawalRule(config, withdrawal_rate=0.05)

        # Create mock portfolio result with fixed 5% returns
        portfolio_balances = np.zeros((5, 1))
        portfolio_balances[0, 0] = 100000  # Set initial balance
        portfolio_returns = np.full((4, 1), 0.05)  # 5% return each year
        asset_balances = np.zeros((1, 5, 1))
        asset_balances[0, 0, 0] = 100000  # Set initial asset balance
        rebalancing_events = np.zeros((5, 1), dtype=bool)
        transaction_costs = np.zeros((5, 1))
        target_weights = np.array([1.0])
        asset_names = ["Portfolio"]

        portfolio_result = PortfolioEvolutionResult(
            portfolio_balances=portfolio_balances,
            asset_balances=asset_balances,
            rebalancing_events=rebalancing_events,
            transaction_costs=transaction_costs,
            portfolio_returns=portfolio_returns,
            target_weights=target_weights,
            asset_names=asset_names,
        )

        # Apply withdrawals
        result = rule.apply_withdrawals(portfolio_result)

        # Verify deterministic results
        # Year 0: Start with 100,000, withdraw 5,000, balance = 95,000
        assert np.isclose(result.portfolio_balances[0, 0], 95000.0)
        assert np.isclose(result.withdrawals[0, 0], 5000.0)

        # Year 1: 95,000 * 1.05 = 99,750, withdraw 4,987.5, balance = 94,762.5
        assert np.isclose(result.portfolio_balances[1, 0], 94762.5)
        assert np.isclose(result.withdrawals[1, 0], 4987.5)

        # Should not fail in this scenario
        assert result.success_rate == 1.0


class TestVPWWithdrawalRule:
    """Test VPWWithdrawalRule."""

    def test_vpw_rate_calculation(self):
        """Test VPW rate calculation."""
        config = WithdrawalRuleConfig(
            initial_balance=100000,
            years=35,  # Age 65 to 100
            num_paths=1,
        )
        rule = VPWWithdrawalRule(
            config, initial_age=65, max_age=100, base_withdrawal_rate=0.04
        )

        # Check that rates increase over time
        assert rule.vpw_rates[0] == 0.04  # Initial rate
        assert rule.vpw_rates[-1] == 1.0  # Final rate should be 1.0

        # Rates should be monotonically increasing
        for i in range(1, len(rule.vpw_rates)):
            assert rule.vpw_rates[i] >= rule.vpw_rates[i - 1]

    def test_withdrawal_calculation(self):
        """Test VPW withdrawal calculation."""
        config = WithdrawalRuleConfig(
            initial_balance=100000,
            years=35,
            num_paths=1,
        )
        rule = VPWWithdrawalRule(
            config, initial_age=65, max_age=100, base_withdrawal_rate=0.04
        )

        # Year 0: 4% of 100,000 = 4,000
        withdrawal = rule.calculate_withdrawal(0, 100000, 0)
        assert withdrawal == 4000.0

        # Year 1: Should be slightly higher rate
        withdrawal = rule.calculate_withdrawal(1, 100000, 0)
        assert withdrawal > 4000.0

    def test_deterministic_no_volatility(self):
        """Test deterministic case with no volatility (toy case)."""
        config = WithdrawalRuleConfig(
            initial_balance=100000,
            years=5,
            num_paths=1,
        )
        rule = VPWWithdrawalRule(
            config, initial_age=65, max_age=100, base_withdrawal_rate=0.04
        )

        # Create mock portfolio result with fixed 5% returns
        portfolio_balances = np.zeros((5, 1))
        portfolio_balances[0, 0] = 100000  # Set initial balance
        portfolio_returns = np.full((4, 1), 0.05)  # 5% return each year
        asset_balances = np.zeros((1, 5, 1))
        asset_balances[0, 0, 0] = 100000  # Set initial asset balance
        rebalancing_events = np.zeros((5, 1), dtype=bool)
        transaction_costs = np.zeros((5, 1))
        target_weights = np.array([1.0])
        asset_names = ["Portfolio"]

        portfolio_result = PortfolioEvolutionResult(
            portfolio_balances=portfolio_balances,
            asset_balances=asset_balances,
            rebalancing_events=rebalancing_events,
            transaction_costs=transaction_costs,
            portfolio_returns=portfolio_returns,
            target_weights=target_weights,
            asset_names=asset_names,
        )

        # Apply withdrawals
        result = rule.apply_withdrawals(portfolio_result)

        # Verify deterministic results
        # Year 0: Start with 100,000, withdraw 4,000, balance = 96,000
        assert np.isclose(result.portfolio_balances[0, 0], 96000.0)
        assert np.isclose(result.withdrawals[0, 0], 4000.0)

        # Should not fail in this scenario
        assert result.success_rate == 1.0


class TestWithdrawalFailureScenarios:
    """Test withdrawal failure scenarios."""

    def test_insufficient_funds_failure(self):
        """Test failure when withdrawal exceeds available balance."""
        config = WithdrawalRuleConfig(
            initial_balance=100000,
            years=3,
            num_paths=1,
        )
        rule = FixedRealWithdrawalRule(
            config, initial_withdrawal_rate=0.5
        )  # 50% withdrawal

        # Create mock portfolio result with negative returns
        portfolio_balances = np.zeros((3, 1))
        portfolio_returns = np.full((2, 1), -0.1)  # -10% return each year
        asset_balances = np.zeros((1, 3, 1))
        rebalancing_events = np.zeros((3, 1), dtype=bool)
        transaction_costs = np.zeros((3, 1))
        target_weights = np.array([1.0])
        asset_names = ["Portfolio"]

        portfolio_result = PortfolioEvolutionResult(
            portfolio_balances=portfolio_balances,
            asset_balances=asset_balances,
            rebalancing_events=rebalancing_events,
            transaction_costs=transaction_costs,
            portfolio_returns=portfolio_returns,
            target_weights=target_weights,
            asset_names=asset_names,
        )

        # Apply withdrawals
        result = rule.apply_withdrawals(portfolio_result)

        # Should fail due to insufficient funds
        assert result.success_rate == 0.0
        assert result.first_failure_years[0] >= 0  # Should fail in some year

    def test_portfolio_depletion(self):
        """Test portfolio depletion scenario."""
        config = WithdrawalRuleConfig(
            initial_balance=100000,
            years=5,
            num_paths=1,
        )
        rule = FixedRealWithdrawalRule(
            config, initial_withdrawal_rate=0.15
        )  # 15% withdrawal

        # Create mock portfolio result with zero returns
        portfolio_balances = np.zeros((5, 1))
        portfolio_returns = np.zeros((4, 1))  # 0% return each year
        asset_balances = np.zeros((1, 5, 1))
        rebalancing_events = np.zeros((5, 1), dtype=bool)
        transaction_costs = np.zeros((5, 1))
        target_weights = np.array([1.0])
        asset_names = ["Portfolio"]

        portfolio_result = PortfolioEvolutionResult(
            portfolio_balances=portfolio_balances,
            asset_balances=asset_balances,
            rebalancing_events=rebalancing_events,
            transaction_costs=transaction_costs,
            portfolio_returns=portfolio_returns,
            target_weights=target_weights,
            asset_names=asset_names,
        )

        # Apply withdrawals
        result = rule.apply_withdrawals(portfolio_result)

        # Should fail due to portfolio depletion
        assert result.success_rate == 0.0
        assert result.first_failure_years[0] >= 0


class TestWithdrawalStatistics:
    """Test withdrawal statistics calculation."""

    def test_statistics_calculation(self):
        """Test withdrawal statistics calculation."""
        config = WithdrawalRuleConfig(
            initial_balance=100000,
            years=5,
            num_paths=2,
        )
        rule = FixedRealWithdrawalRule(config, initial_withdrawal_rate=0.04)

        # Create mock portfolio result
        portfolio_balances = np.zeros((5, 2))
        portfolio_balances[0, :] = 100000  # Set initial balance for both paths
        portfolio_returns = np.full((4, 2), 0.05)  # 5% return each year
        asset_balances = np.zeros((1, 5, 2))
        asset_balances[0, 0, :] = 100000  # Set initial asset balance for both paths
        rebalancing_events = np.zeros((5, 2), dtype=bool)
        transaction_costs = np.zeros((5, 2))
        target_weights = np.array([1.0])
        asset_names = ["Portfolio"]

        portfolio_result = PortfolioEvolutionResult(
            portfolio_balances=portfolio_balances,
            asset_balances=asset_balances,
            rebalancing_events=rebalancing_events,
            transaction_costs=transaction_costs,
            portfolio_returns=portfolio_returns,
            target_weights=target_weights,
            asset_names=asset_names,
        )

        # Apply withdrawals
        result = rule.apply_withdrawals(portfolio_result)

        # Calculate statistics
        stats = calculate_withdrawal_statistics(result)

        # Verify statistics are calculated
        assert "success_rate" in stats
        assert "final_balance_mean" in stats
        assert "total_withdrawals_mean" in stats
        assert "avg_annual_withdrawal_mean" in stats
        assert "failure_rate" in stats

        # Success rate should be 1.0 for this scenario
        assert stats["success_rate"] == 1.0
        assert stats["failure_rate"] == 0.0


class TestWithdrawalComparison:
    """Test withdrawal strategy comparison."""

    def test_strategy_comparison(self):
        """Test comparison of different withdrawal strategies."""
        config = WithdrawalRuleConfig(
            initial_balance=100000,
            years=5,
            num_paths=1,
        )

        # Create different withdrawal rules
        fixed_real = FixedRealWithdrawalRule(config, initial_withdrawal_rate=0.04)
        fixed_percentage = FixedPercentageWithdrawalRule(config, withdrawal_rate=0.05)
        vpw = VPWWithdrawalRule(
            config, initial_age=65, max_age=100, base_withdrawal_rate=0.04
        )

        # Create mock portfolio result
        portfolio_balances = np.zeros((5, 1))
        portfolio_balances[0, 0] = 100000  # Set initial balance
        portfolio_returns = np.full((4, 1), 0.05)  # 5% return each year
        asset_balances = np.zeros((1, 5, 1))
        asset_balances[0, 0, 0] = 100000  # Set initial asset balance
        rebalancing_events = np.zeros((5, 1), dtype=bool)
        transaction_costs = np.zeros((5, 1))
        target_weights = np.array([1.0])
        asset_names = ["Portfolio"]

        portfolio_result = PortfolioEvolutionResult(
            portfolio_balances=portfolio_balances,
            asset_balances=asset_balances,
            rebalancing_events=rebalancing_events,
            transaction_costs=transaction_costs,
            portfolio_returns=portfolio_returns,
            target_weights=target_weights,
            asset_names=asset_names,
        )

        # Compare strategies
        comparison = compare_withdrawal_strategies(
            portfolio_result, [fixed_real, fixed_percentage, vpw]
        )

        # Verify comparison results
        assert len(comparison) == 3
        assert "strategy_0" in comparison
        assert "strategy_1" in comparison
        assert "strategy_2" in comparison

        # All strategies should have success rate of 1.0 for this scenario
        for strategy in comparison.values():
            assert strategy["success_rate"] == 1.0


class TestWithdrawalValidation:
    """Test withdrawal rule validation."""

    def test_validation_success(self):
        """Test successful validation."""
        config = WithdrawalRuleConfig(
            initial_balance=100000,
            years=5,
            num_paths=1,
        )
        rule = FixedRealWithdrawalRule(config, initial_withdrawal_rate=0.04)

        # Create mock portfolio result
        portfolio_balances = np.zeros((5, 1))
        portfolio_balances[0, 0] = 100000  # Set initial balance
        portfolio_returns = np.full((4, 1), 0.05)  # 5% return each year
        asset_balances = np.zeros((1, 5, 1))
        asset_balances[0, 0, 0] = 100000  # Set initial asset balance
        rebalancing_events = np.zeros((5, 1), dtype=bool)
        transaction_costs = np.zeros((5, 1))
        target_weights = np.array([1.0])
        asset_names = ["Portfolio"]

        portfolio_result = PortfolioEvolutionResult(
            portfolio_balances=portfolio_balances,
            asset_balances=asset_balances,
            rebalancing_events=rebalancing_events,
            transaction_costs=transaction_costs,
            portfolio_returns=portfolio_returns,
            target_weights=target_weights,
            asset_names=asset_names,
        )

        # Validate with expected properties
        expected_properties = {"success_rate": 1.0}
        is_valid, message = validate_withdrawal_rule(
            rule, portfolio_result, expected_properties
        )

        assert is_valid
        assert "within tolerance" in message

    def test_validation_failure(self):
        """Test validation failure."""
        config = WithdrawalRuleConfig(
            initial_balance=100000,
            years=5,
            num_paths=1,
        )
        rule = FixedRealWithdrawalRule(config, initial_withdrawal_rate=0.04)

        # Create mock portfolio result
        portfolio_balances = np.zeros((5, 1))
        portfolio_balances[0, 0] = 100000  # Set initial balance
        portfolio_returns = np.full((4, 1), 0.05)  # 5% return each year
        asset_balances = np.zeros((1, 5, 1))
        asset_balances[0, 0, 0] = 100000  # Set initial asset balance
        rebalancing_events = np.zeros((5, 1), dtype=bool)
        transaction_costs = np.zeros((5, 1))
        target_weights = np.array([1.0])
        asset_names = ["Portfolio"]

        portfolio_result = PortfolioEvolutionResult(
            portfolio_balances=portfolio_balances,
            asset_balances=asset_balances,
            rebalancing_events=rebalancing_events,
            transaction_costs=transaction_costs,
            portfolio_returns=portfolio_returns,
            target_weights=target_weights,
            asset_names=asset_names,
        )

        # Validate with incorrect expected properties
        expected_properties = {"success_rate": 0.0}  # Should be 1.0
        is_valid, message = validate_withdrawal_rule(
            rule, portfolio_result, expected_properties
        )

        assert not is_valid
        assert "expected" in message and "got" in message


class TestIntegrationWithPortfolioEvolution:
    """Test integration with portfolio evolution."""

    def test_integration_with_real_portfolio(self):
        """Test integration with real portfolio evolution."""
        # Create asset classes
        asset_classes = [
            AssetClass(
                name="Stocks", expected_return=0.07, volatility=0.18, weight=0.6
            ),
            AssetClass(name="Bonds", expected_return=0.03, volatility=0.06, weight=0.4),
        ]

        # Create portfolio evolution config
        from app.models.portfolio_evolution import (
            PortfolioEvolution,
            PortfolioEvolutionConfig,
        )

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

        # Test different withdrawal rules
        fixed_real = FixedRealWithdrawalRule(
            withdrawal_config, initial_withdrawal_rate=0.04
        )
        fixed_percentage = FixedPercentageWithdrawalRule(
            withdrawal_config, withdrawal_rate=0.05
        )

        # Apply withdrawal rules
        fixed_real_result = fixed_real.apply_withdrawals(portfolio_result)
        fixed_percentage_result = fixed_percentage.apply_withdrawals(portfolio_result)

        # Verify results
        assert fixed_real_result.success_rate >= 0.0
        assert fixed_real_result.success_rate <= 1.0
        assert fixed_percentage_result.success_rate >= 0.0
        assert fixed_percentage_result.success_rate <= 1.0

        # Verify shapes
        assert fixed_real_result.withdrawals.shape == (10, 100)
        assert fixed_real_result.portfolio_balances.shape == (10, 100)
        assert fixed_real_result.failures.shape == (10, 100)
        assert fixed_real_result.first_failure_years.shape == (100,)
