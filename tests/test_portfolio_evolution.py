"""
Tests for portfolio evolution module.
"""

import numpy as np
import pytest

from app.models.portfolio_evolution import (
    PortfolioEvolution,
    PortfolioEvolutionConfig,
    PortfolioEvolutionResult,
    calculate_portfolio_statistics,
    compare_rebalancing_strategies,
    validate_rebalancing_accuracy,
)
from app.models.random_returns import AssetClass, RandomReturnsConfig


class TestPortfolioEvolutionConfig:
    """Test PortfolioEvolutionConfig validation."""

    def test_valid_config(self):
        """Test valid configuration."""
        asset_classes = [
            AssetClass(
                name="Stocks", expected_return=0.07, volatility=0.18, weight=0.6
            ),
            AssetClass(name="Bonds", expected_return=0.03, volatility=0.06, weight=0.4),
        ]

        config = PortfolioEvolutionConfig(
            initial_balance=100000,
            asset_classes=asset_classes,
            years=10,
            num_paths=1000,
        )

        assert config.initial_balance == 100000
        assert len(config.asset_classes) == 2
        assert config.years == 10
        assert config.num_paths == 1000
        assert config.rebalance_threshold == 0.05
        assert config.transaction_cost_rate == 0.0
        assert config.enable_rebalancing is True

    def test_invalid_weights(self):
        """Test invalid portfolio weights."""
        asset_classes = [
            AssetClass(
                name="Stocks", expected_return=0.07, volatility=0.18, weight=0.6
            ),
            AssetClass(
                name="Bonds", expected_return=0.03, volatility=0.06, weight=0.5
            ),  # Total > 1.0
        ]

        with pytest.raises(ValueError, match="Portfolio weights must sum to 1.0"):
            PortfolioEvolutionConfig(
                initial_balance=100000,
                asset_classes=asset_classes,
                years=10,
                num_paths=1000,
            )

    def test_invalid_initial_balance(self):
        """Test invalid initial balance."""
        asset_classes = [
            AssetClass(
                name="Stocks", expected_return=0.07, volatility=0.18, weight=1.0
            ),
        ]

        with pytest.raises(Exception):  # Pydantic validation error
            PortfolioEvolutionConfig(
                initial_balance=0,
                asset_classes=asset_classes,
                years=10,
                num_paths=1000,
            )

    def test_invalid_years(self):
        """Test invalid number of years."""
        asset_classes = [
            AssetClass(
                name="Stocks", expected_return=0.07, volatility=0.18, weight=1.0
            ),
        ]

        with pytest.raises(Exception):  # Pydantic validation error
            PortfolioEvolutionConfig(
                initial_balance=100000,
                asset_classes=asset_classes,
                years=0,
                num_paths=1000,
            )

    def test_invalid_num_paths(self):
        """Test invalid number of paths."""
        asset_classes = [
            AssetClass(
                name="Stocks", expected_return=0.07, volatility=0.18, weight=1.0
            ),
        ]

        with pytest.raises(Exception):  # Pydantic validation error
            PortfolioEvolutionConfig(
                initial_balance=100000,
                asset_classes=asset_classes,
                years=10,
                num_paths=0,
            )

    def test_empty_asset_classes(self):
        """Test empty asset classes."""
        with pytest.raises(Exception):  # Pydantic validation error
            PortfolioEvolutionConfig(
                initial_balance=100000,
                asset_classes=[],
                years=10,
                num_paths=1000,
            )


class TestPortfolioEvolution:
    """Test PortfolioEvolution simulation."""

    def test_single_asset_no_rebalancing(self):
        """Test single asset portfolio with no rebalancing."""
        asset_classes = [
            AssetClass(
                name="Stocks", expected_return=0.07, volatility=0.18, weight=1.0
            ),
        ]

        config = PortfolioEvolutionConfig(
            initial_balance=100000,
            asset_classes=asset_classes,
            years=2,
            num_paths=1,
            enable_rebalancing=False,
            seed=42,
        )

        evolution = PortfolioEvolution(config)
        result = evolution.simulate()

        # Check dimensions
        assert result.portfolio_balances.shape == (2, 1)
        assert result.asset_balances.shape == (1, 2, 1)
        assert result.rebalancing_events.shape == (2, 1)
        assert result.transaction_costs.shape == (2, 1)
        assert result.portfolio_returns.shape == (2, 1)

        # Check initial values
        assert result.portfolio_balances[0, 0] == 100000
        assert result.asset_balances[0, 0, 0] == 100000

        # Check no rebalancing occurred
        assert not np.any(result.rebalancing_events)
        assert np.all(result.transaction_costs == 0)

    def test_two_asset_rebalancing(self):
        """Test two asset portfolio with rebalancing."""
        asset_classes = [
            AssetClass(
                name="Stocks", expected_return=0.07, volatility=0.18, weight=0.6
            ),
            AssetClass(name="Bonds", expected_return=0.03, volatility=0.06, weight=0.4),
        ]

        config = PortfolioEvolutionConfig(
            initial_balance=100000,
            asset_classes=asset_classes,
            years=3,
            num_paths=1,
            rebalance_threshold=0.01,  # Very low threshold to trigger rebalancing
            seed=42,
        )

        evolution = PortfolioEvolution(config)
        result = evolution.simulate()

        # Check dimensions
        assert result.portfolio_balances.shape == (3, 1)
        assert result.asset_balances.shape == (2, 3, 1)
        assert result.rebalancing_events.shape == (3, 1)
        assert result.transaction_costs.shape == (3, 1)
        assert result.portfolio_returns.shape == (3, 1)

        # Check initial allocation
        assert result.asset_balances[0, 0, 0] == 60000  # 60% stocks
        assert result.asset_balances[1, 0, 0] == 40000  # 40% bonds

        # Check target weights
        assert np.allclose(result.target_weights, [0.6, 0.4])

    def test_deterministic_returns(self):
        """Test with deterministic returns (zero volatility)."""
        asset_classes = [
            AssetClass(name="Stocks", expected_return=0.05, volatility=0.0, weight=0.6),
            AssetClass(name="Bonds", expected_return=0.03, volatility=0.0, weight=0.4),
        ]

        config = PortfolioEvolutionConfig(
            initial_balance=100000,
            asset_classes=asset_classes,
            years=2,
            num_paths=1,
            seed=42,
        )

        evolution = PortfolioEvolution(config)
        result = evolution.simulate()

        # With zero volatility, returns should be deterministic
        expected_stock_return = 0.05
        expected_bond_return = 0.03
        expected_portfolio_return = (
            0.6 * expected_stock_return + 0.4 * expected_bond_return
        )

        # Check portfolio return
        assert np.isclose(
            result.portfolio_returns[0, 0], expected_portfolio_return, atol=1e-10
        )

        # Check final balance
        expected_final_balance = 100000 * (1 + expected_portfolio_return)
        assert np.isclose(
            result.portfolio_balances[1, 0], expected_final_balance, atol=1e-10
        )

    def test_transaction_costs(self):
        """Test transaction costs are applied correctly."""
        asset_classes = [
            AssetClass(
                name="Stocks", expected_return=0.07, volatility=0.18, weight=0.6
            ),
            AssetClass(name="Bonds", expected_return=0.03, volatility=0.06, weight=0.4),
        ]

        config = PortfolioEvolutionConfig(
            initial_balance=100000,
            asset_classes=asset_classes,
            years=2,
            num_paths=1,
            rebalance_threshold=0.01,
            transaction_cost_rate=0.001,  # 0.1% transaction cost
            seed=42,
        )

        evolution = PortfolioEvolution(config)
        result = evolution.simulate()

        # Check that transaction costs are calculated when rebalancing occurs
        if np.any(result.rebalancing_events):
            assert np.any(result.transaction_costs > 0)

    def test_no_rebalancing_when_disabled(self):
        """Test that no rebalancing occurs when disabled."""
        asset_classes = [
            AssetClass(
                name="Stocks", expected_return=0.07, volatility=0.18, weight=0.6
            ),
            AssetClass(name="Bonds", expected_return=0.03, volatility=0.06, weight=0.4),
        ]

        config = PortfolioEvolutionConfig(
            initial_balance=100000,
            asset_classes=asset_classes,
            years=3,
            num_paths=1,
            enable_rebalancing=False,
            seed=42,
        )

        evolution = PortfolioEvolution(config)
        result = evolution.simulate()

        # Check no rebalancing occurred
        assert not np.any(result.rebalancing_events)
        assert np.all(result.transaction_costs == 0)

    def test_multiple_paths(self):
        """Test simulation with multiple paths."""
        asset_classes = [
            AssetClass(
                name="Stocks", expected_return=0.07, volatility=0.18, weight=1.0
            ),
        ]

        config = PortfolioEvolutionConfig(
            initial_balance=100000,
            asset_classes=asset_classes,
            years=2,
            num_paths=100,
            seed=42,
        )

        evolution = PortfolioEvolution(config)
        result = evolution.simulate()

        # Check dimensions
        assert result.portfolio_balances.shape == (2, 100)
        assert result.asset_balances.shape == (1, 2, 100)
        assert result.rebalancing_events.shape == (2, 100)
        assert result.transaction_costs.shape == (2, 100)
        assert result.portfolio_returns.shape == (2, 100)

        # Check all paths start with same initial balance
        assert np.all(result.portfolio_balances[0, :] == 100000)
        assert np.all(result.asset_balances[0, 0, :] == 100000)

    def test_custom_returns_config(self):
        """Test with custom returns configuration."""
        asset_classes = [
            AssetClass(
                name="Stocks", expected_return=0.07, volatility=0.18, weight=0.6
            ),
            AssetClass(name="Bonds", expected_return=0.03, volatility=0.06, weight=0.4),
        ]

        returns_config = RandomReturnsConfig(
            asset_classes=asset_classes,
            years=2,
            num_paths=1,
            distribution="lognormal",
            seed=42,
        )

        config = PortfolioEvolutionConfig(
            initial_balance=100000,
            asset_classes=asset_classes,
            years=2,
            num_paths=1,
            returns_config=returns_config,
        )

        evolution = PortfolioEvolution(config)
        result = evolution.simulate()

        # Check that custom config was used
        assert result.portfolio_balances.shape == (2, 1)
        assert result.asset_balances.shape == (2, 2, 1)


class TestRebalancingAccuracy:
    """Test rebalancing accuracy validation."""

    def test_perfect_rebalancing(self):
        """Test validation with perfect rebalancing."""
        # Create a result with perfect rebalancing
        portfolio_balances = np.array([[100000, 100000], [105000, 103000]])
        asset_balances = np.array(
            [
                [[60000, 60000], [63000, 61800]],  # Stocks: 60% of portfolio
                [[40000, 40000], [42000, 41200]],  # Bonds: 40% of portfolio
            ]
        )
        rebalancing_events = np.array([[False, False], [True, True]])
        transaction_costs = np.array([[0, 0], [50, 30]])
        portfolio_returns = np.array([[0.05, 0.03], [0, 0]])
        target_weights = np.array([0.6, 0.4])
        asset_names = ["Stocks", "Bonds"]

        result = PortfolioEvolutionResult(
            portfolio_balances=portfolio_balances,
            asset_balances=asset_balances,
            rebalancing_events=rebalancing_events,
            transaction_costs=transaction_costs,
            portfolio_returns=portfolio_returns,
            target_weights=target_weights,
            asset_names=asset_names,
        )

        is_valid, message = validate_rebalancing_accuracy(result, tolerance=1e-6)
        assert is_valid
        assert "within tolerance" in message

    def test_imperfect_rebalancing(self):
        """Test validation with imperfect rebalancing."""
        # Create a result with imperfect rebalancing
        portfolio_balances = np.array([[100000], [105000]])
        asset_balances = np.array(
            [
                [[60000], [65000]],  # Stocks: 61.9% instead of 60%
                [[40000], [40000]],  # Bonds: 38.1% instead of 40%
            ]
        )
        rebalancing_events = np.array(
            [[False], [True]]
        )  # Rebalancing occurred in year 1
        transaction_costs = np.array([[0], [50]])
        portfolio_returns = np.array([[0.05], [0]])
        target_weights = np.array([0.6, 0.4])
        asset_names = ["Stocks", "Bonds"]

        result = PortfolioEvolutionResult(
            portfolio_balances=portfolio_balances,
            asset_balances=asset_balances,
            rebalancing_events=rebalancing_events,
            transaction_costs=transaction_costs,
            portfolio_returns=portfolio_returns,
            target_weights=target_weights,
            asset_names=asset_names,
        )

        is_valid, message = validate_rebalancing_accuracy(result, tolerance=1e-6)
        assert not is_valid
        assert "exceeds tolerance" in message

    def test_depleted_portfolio(self):
        """Test validation with depleted portfolio."""
        # Create a result with depleted portfolio
        portfolio_balances = np.array([[100000], [0]])
        asset_balances = np.array(
            [
                [[60000], [0]],
                [[40000], [0]],
            ]
        )
        rebalancing_events = np.array([[False], [False]])
        transaction_costs = np.array([[0], [0]])
        portfolio_returns = np.array([[-1.0], [0]])
        target_weights = np.array([0.6, 0.4])
        asset_names = ["Stocks", "Bonds"]

        result = PortfolioEvolutionResult(
            portfolio_balances=portfolio_balances,
            asset_balances=asset_balances,
            rebalancing_events=rebalancing_events,
            transaction_costs=transaction_costs,
            portfolio_returns=portfolio_returns,
            target_weights=target_weights,
            asset_names=asset_names,
        )

        is_valid, message = validate_rebalancing_accuracy(result, tolerance=1e-6)
        assert is_valid  # Depleted portfolios should be skipped


class TestPortfolioStatistics:
    """Test portfolio statistics calculation."""

    def test_portfolio_statistics(self):
        """Test portfolio statistics calculation."""
        # Create a simple result
        portfolio_balances = np.array(
            [[100000, 100000], [105000, 103000], [110000, 106000]]
        )
        asset_balances = np.array(
            [
                [[60000, 60000], [63000, 61800], [66000, 63600]],
                [[40000, 40000], [42000, 41200], [44000, 42400]],
            ]
        )
        rebalancing_events = np.array([[False, False], [True, True], [False, False]])
        transaction_costs = np.array([[0, 0], [50, 30], [0, 0]])
        portfolio_returns = np.array([[0.05, 0.03], [0.0476, 0.0291], [0, 0]])
        target_weights = np.array([0.6, 0.4])
        asset_names = ["Stocks", "Bonds"]

        result = PortfolioEvolutionResult(
            portfolio_balances=portfolio_balances,
            asset_balances=asset_balances,
            rebalancing_events=rebalancing_events,
            transaction_costs=transaction_costs,
            portfolio_returns=portfolio_returns,
            target_weights=target_weights,
            asset_names=asset_names,
        )

        stats = calculate_portfolio_statistics(result)

        # Check that all expected statistics are present
        expected_keys = [
            "final_balance_mean",
            "final_balance_std",
            "final_balance_p5",
            "final_balance_p50",
            "final_balance_p95",
            "rebalancing_frequency",
            "total_transaction_costs_mean",
            "total_transaction_costs_std",
            "portfolio_volatility_mean",
            "portfolio_volatility_std",
            "portfolio_mean_return_mean",
            "portfolio_mean_return_std",
        ]

        for key in expected_keys:
            assert key in stats
            assert isinstance(stats[key], (int, float))
            assert not np.isnan(stats[key])

    def test_rebalancing_frequency(self):
        """Test rebalancing frequency calculation."""
        # Create result with known rebalancing pattern
        portfolio_balances = np.array([[100000], [105000], [110000]])
        asset_balances = np.array(
            [
                [[60000], [63000], [66000]],
                [[40000], [42000], [44000]],
            ]
        )
        rebalancing_events = np.array([[False], [True], [False]])  # 1 out of 3 events
        transaction_costs = np.array([[0], [50], [0]])
        portfolio_returns = np.array([[0.05], [0.0476], [0]])
        target_weights = np.array([0.6, 0.4])
        asset_names = ["Stocks", "Bonds"]

        result = PortfolioEvolutionResult(
            portfolio_balances=portfolio_balances,
            asset_balances=asset_balances,
            rebalancing_events=rebalancing_events,
            transaction_costs=transaction_costs,
            portfolio_returns=portfolio_returns,
            target_weights=target_weights,
            asset_names=asset_names,
        )

        stats = calculate_portfolio_statistics(result)

        # Rebalancing frequency should be 1/3 ≈ 0.333
        expected_frequency = 1.0 / 3.0
        assert np.isclose(
            stats["rebalancing_frequency"], expected_frequency, atol=1e-10
        )


class TestRebalancingComparison:
    """Test rebalancing strategy comparison."""

    def test_compare_rebalancing_strategies(self):
        """Test comparison of different rebalancing strategies."""
        asset_classes = [
            AssetClass(
                name="Stocks", expected_return=0.07, volatility=0.18, weight=0.6
            ),
            AssetClass(name="Bonds", expected_return=0.03, volatility=0.06, weight=0.4),
        ]

        config = PortfolioEvolutionConfig(
            initial_balance=100000,
            asset_classes=asset_classes,
            years=2,
            num_paths=10,  # Small number for fast test
            seed=42,
        )

        thresholds = [0.01, 0.05, 0.10]
        results = compare_rebalancing_strategies(config, thresholds)

        # Check that results exist for all thresholds
        for threshold in thresholds:
            key = f"threshold_{threshold}"
            assert key in results

            # Check that all expected statistics are present
            stats = results[key]
            expected_keys = [
                "final_balance_mean",
                "rebalancing_frequency",
                "total_transaction_costs_mean",
            ]

            for expected_key in expected_keys:
                assert expected_key in stats
                assert isinstance(stats[expected_key], (int, float))
                assert not np.isnan(stats[expected_key])

    def test_rebalancing_threshold_impact(self):
        """Test that different thresholds produce different results."""
        asset_classes = [
            AssetClass(
                name="Stocks", expected_return=0.07, volatility=0.18, weight=0.6
            ),
            AssetClass(name="Bonds", expected_return=0.03, volatility=0.06, weight=0.4),
        ]

        config = PortfolioEvolutionConfig(
            initial_balance=100000,
            asset_classes=asset_classes,
            years=3,
            num_paths=100,
            seed=42,
        )

        # Compare very tight vs very loose rebalancing
        tight_config = config.model_copy()
        tight_config.rebalance_threshold = 0.001

        loose_config = config.model_copy()
        loose_config.rebalance_threshold = 0.5

        tight_evolution = PortfolioEvolution(tight_config)
        loose_evolution = PortfolioEvolution(loose_config)

        tight_result = tight_evolution.simulate()
        loose_result = loose_evolution.simulate()

        # Tight rebalancing should have more rebalancing events
        tight_frequency = np.mean(tight_result.rebalancing_events)
        loose_frequency = np.mean(loose_result.rebalancing_events)

        assert tight_frequency > loose_frequency


class TestIntegration:
    """Integration tests for portfolio evolution."""

    def test_end_to_end_simulation(self):
        """Test complete end-to-end simulation."""
        asset_classes = [
            AssetClass(
                name="Stocks", expected_return=0.07, volatility=0.18, weight=0.6
            ),
            AssetClass(name="Bonds", expected_return=0.03, volatility=0.06, weight=0.4),
        ]

        config = PortfolioEvolutionConfig(
            initial_balance=100000,
            asset_classes=asset_classes,
            years=10,
            num_paths=1000,
            rebalance_threshold=0.05,
            transaction_cost_rate=0.001,
            seed=42,
        )

        evolution = PortfolioEvolution(config)
        result = evolution.simulate()

        # Validate rebalancing accuracy with reasonable tolerance
        is_valid, message = validate_rebalancing_accuracy(result, tolerance=0.01)
        assert is_valid, f"Rebalancing validation failed: {message}"

        # Calculate statistics
        stats = calculate_portfolio_statistics(result)

        # Check reasonable statistics
        assert stats["final_balance_mean"] > 0
        assert stats["rebalancing_frequency"] >= 0
        assert stats["rebalancing_frequency"] <= 1
        assert stats["total_transaction_costs_mean"] >= 0

        # Check that portfolio grew on average (positive expected returns)
        assert stats["final_balance_mean"] > 100000

    def test_three_asset_portfolio(self):
        """Test three asset portfolio simulation."""
        asset_classes = [
            AssetClass(
                name="Stocks", expected_return=0.07, volatility=0.18, weight=0.5
            ),
            AssetClass(name="Bonds", expected_return=0.03, volatility=0.06, weight=0.3),
            AssetClass(name="REITs", expected_return=0.06, volatility=0.15, weight=0.2),
        ]

        config = PortfolioEvolutionConfig(
            initial_balance=100000,
            asset_classes=asset_classes,
            years=5,
            num_paths=100,
            rebalance_threshold=0.05,
            seed=42,
        )

        evolution = PortfolioEvolution(config)
        result = evolution.simulate()

        # Check dimensions
        assert result.asset_balances.shape[0] == 3  # Three assets
        assert result.target_weights.shape[0] == 3
        assert len(result.asset_names) == 3

        # Check target weights sum to 1
        assert np.isclose(np.sum(result.target_weights), 1.0, atol=1e-10)

        # Validate rebalancing with reasonable tolerance
        is_valid, message = validate_rebalancing_accuracy(result, tolerance=0.01)
        assert is_valid, f"Rebalancing validation failed: {message}"
