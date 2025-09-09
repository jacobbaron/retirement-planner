"""
Tests for random returns generator module.
"""

import numpy as np
import pytest

from app.models.random_returns import (
    AssetClass,
    RandomReturnsConfig,
    RandomReturnsGenerator,
    calculate_annualized_returns,
    calculate_cumulative_returns,
    calculate_portfolio_returns,
    create_default_portfolio,
    create_three_asset_portfolio,
    validate_returns_statistics,
)


class TestAssetClass:
    """Test AssetClass model."""

    def test_valid_asset_class(self):
        """Test creating a valid asset class."""
        asset = AssetClass(
            name="Stocks", expected_return=0.07, volatility=0.18, weight=0.6
        )
        assert asset.name == "Stocks"
        assert asset.expected_return == 0.07
        assert asset.volatility == 0.18
        assert asset.weight == 0.6

    def test_invalid_expected_return(self):
        """Test validation of expected return bounds."""
        with pytest.raises(ValueError):
            AssetClass(
                name="Stocks",
                expected_return=-1.5,  # Too low
                volatility=0.18,
                weight=0.6,
            )

        with pytest.raises(ValueError):
            AssetClass(
                name="Stocks",
                expected_return=2.5,  # Too high
                volatility=0.18,
                weight=0.6,
            )

    def test_invalid_volatility(self):
        """Test validation of volatility bounds."""
        with pytest.raises(ValueError):
            AssetClass(
                name="Stocks",
                expected_return=0.07,
                volatility=-0.1,  # Negative volatility
                weight=0.6,
            )

    def test_invalid_weight(self):
        """Test validation of weight bounds."""
        with pytest.raises(ValueError):
            AssetClass(
                name="Stocks",
                expected_return=0.07,
                volatility=0.18,
                weight=1.5,  # Weight > 1
            )


class TestRandomReturnsConfig:
    """Test RandomReturnsConfig model."""

    def test_valid_config(self):
        """Test creating a valid configuration."""
        assets = create_default_portfolio()
        config = RandomReturnsConfig(
            asset_classes=assets,
            years=30,
            num_paths=1000,
            distribution="normal",
            seed=42,
        )
        assert len(config.asset_classes) == 2
        assert config.years == 30
        assert config.num_paths == 1000
        assert config.distribution == "normal"
        assert config.seed == 42

    def test_invalid_portfolio_weights(self):
        """Test validation of portfolio weights."""
        assets = [
            AssetClass(
                name="Stocks", expected_return=0.07, volatility=0.18, weight=0.6
            ),
            AssetClass(
                name="Bonds", expected_return=0.03, volatility=0.06, weight=0.5
            ),  # Total > 1
        ]

        with pytest.raises(ValueError, match="Portfolio weights must sum to 1.0"):
            RandomReturnsConfig(asset_classes=assets, years=30, num_paths=1000)

    def test_invalid_years(self):
        """Test validation of years parameter."""
        assets = create_default_portfolio()

        with pytest.raises(ValueError):
            RandomReturnsConfig(
                asset_classes=assets, years=0, num_paths=1000  # Invalid
            )

    def test_invalid_num_paths(self):
        """Test validation of num_paths parameter."""
        assets = create_default_portfolio()

        with pytest.raises(ValueError):
            RandomReturnsConfig(asset_classes=assets, years=30, num_paths=0)  # Invalid


class TestRandomReturnsGenerator:
    """Test RandomReturnsGenerator class."""

    def test_initialization(self):
        """Test generator initialization."""
        assets = create_default_portfolio()
        config = RandomReturnsConfig(
            asset_classes=assets, years=30, num_paths=1000, seed=42
        )
        generator = RandomReturnsGenerator(config)
        assert generator.config == config

    def test_generate_normal_returns(self):
        """Test generating normal returns."""
        assets = create_default_portfolio()
        config = RandomReturnsConfig(
            asset_classes=assets,
            years=10,
            num_paths=1000,
            distribution="normal",
            seed=42,
        )
        generator = RandomReturnsGenerator(config)
        returns = generator.generate_returns()

        # Check shape
        assert returns.shape == (2, 10, 1000)  # 2 assets, 10 years, 1000 paths

        # Check that returns are reasonable
        assert np.all(np.isfinite(returns))

        # Check statistics are close to expected values
        is_valid, message = validate_returns_statistics(
            returns,
            generator.get_expected_returns(),
            generator.get_volatilities(),
            tolerance=0.05,  # 5% tolerance for small sample
        )
        assert is_valid, message

    def test_generate_lognormal_returns(self):
        """Test generating lognormal returns."""
        assets = create_default_portfolio()
        config = RandomReturnsConfig(
            asset_classes=assets,
            years=10,
            num_paths=1000,
            distribution="lognormal",
            seed=42,
        )
        generator = RandomReturnsGenerator(config)
        returns = generator.generate_returns()

        # Check shape
        assert returns.shape == (2, 10, 1000)

        # Check that returns are finite and >= -1 (can't lose more than 100%)
        assert np.all(np.isfinite(returns))
        assert np.all(returns >= -1.0)

        # Check statistics are close to expected values
        is_valid, message = validate_returns_statistics(
            returns,
            generator.get_expected_returns(),
            generator.get_volatilities(),
            tolerance=0.05,
        )
        assert is_valid, message

    def test_deterministic_seeding(self):
        """Test that seeding produces reproducible results."""
        assets = create_default_portfolio()
        config = RandomReturnsConfig(
            asset_classes=assets, years=5, num_paths=100, seed=123
        )

        generator1 = RandomReturnsGenerator(config)
        returns1 = generator1.generate_returns()

        generator2 = RandomReturnsGenerator(config)
        returns2 = generator2.generate_returns()

        # Results should be identical with same seed
        np.testing.assert_array_equal(returns1, returns2)

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        assets = create_default_portfolio()

        config1 = RandomReturnsConfig(
            asset_classes=assets, years=5, num_paths=100, seed=123
        )

        config2 = RandomReturnsConfig(
            asset_classes=assets, years=5, num_paths=100, seed=456
        )

        generator1 = RandomReturnsGenerator(config1)
        returns1 = generator1.generate_returns()

        generator2 = RandomReturnsGenerator(config2)
        returns2 = generator2.generate_returns()

        # Results should be different with different seeds
        assert not np.array_equal(returns1, returns2)

    def test_get_asset_names(self):
        """Test getting asset names."""
        assets = create_three_asset_portfolio()
        config = RandomReturnsConfig(asset_classes=assets, years=10, num_paths=1000)
        generator = RandomReturnsGenerator(config)

        names = generator.get_asset_names()
        assert names == ["Stocks", "Bonds", "REITs"]

    def test_get_asset_weights(self):
        """Test getting asset weights."""
        assets = create_three_asset_portfolio()
        config = RandomReturnsConfig(asset_classes=assets, years=10, num_paths=1000)
        generator = RandomReturnsGenerator(config)

        weights = generator.get_asset_weights()
        expected_weights = np.array([0.5, 0.3, 0.2])
        np.testing.assert_array_equal(weights, expected_weights)

    def test_get_expected_returns(self):
        """Test getting expected returns."""
        assets = create_three_asset_portfolio()
        config = RandomReturnsConfig(asset_classes=assets, years=10, num_paths=1000)
        generator = RandomReturnsGenerator(config)

        expected_returns = generator.get_expected_returns()
        expected_values = np.array([0.07, 0.03, 0.06])
        np.testing.assert_array_equal(expected_returns, expected_values)

    def test_get_volatilities(self):
        """Test getting volatilities."""
        assets = create_three_asset_portfolio()
        config = RandomReturnsConfig(asset_classes=assets, years=10, num_paths=1000)
        generator = RandomReturnsGenerator(config)

        volatilities = generator.get_volatilities()
        expected_values = np.array([0.18, 0.06, 0.15])
        np.testing.assert_array_equal(volatilities, expected_values)


class TestPortfolioFunctions:
    """Test portfolio utility functions."""

    def test_create_default_portfolio(self):
        """Test creating default portfolio."""
        portfolio = create_default_portfolio()
        assert len(portfolio) == 2
        assert portfolio[0].name == "Stocks"
        assert portfolio[1].name == "Bonds"
        assert sum(asset.weight for asset in portfolio) == 1.0

    def test_create_three_asset_portfolio(self):
        """Test creating three-asset portfolio."""
        portfolio = create_three_asset_portfolio()
        assert len(portfolio) == 3
        assert portfolio[0].name == "Stocks"
        assert portfolio[1].name == "Bonds"
        assert portfolio[2].name == "REITs"
        assert sum(asset.weight for asset in portfolio) == 1.0

    def test_calculate_portfolio_returns(self):
        """Test calculating portfolio returns."""
        # Create simple test data
        asset_returns = np.array(
            [
                [[0.1, 0.2], [0.05, 0.15]],  # Asset 1: 2 years, 2 paths
                [[0.03, 0.04], [0.02, 0.03]],  # Asset 2: 2 years, 2 paths
            ]
        )
        weights = np.array([0.6, 0.4])

        portfolio_returns = calculate_portfolio_returns(asset_returns, weights)

        # Expected portfolio returns
        expected = np.array(
            [
                [0.6 * 0.1 + 0.4 * 0.03, 0.6 * 0.2 + 0.4 * 0.04],  # Year 1
                [0.6 * 0.05 + 0.4 * 0.02, 0.6 * 0.15 + 0.4 * 0.03],  # Year 2
            ]
        )

        np.testing.assert_array_almost_equal(portfolio_returns, expected)

    def test_calculate_cumulative_returns(self):
        """Test calculating cumulative returns."""
        # Simple test case: 10% return each year for 2 years
        returns = np.array([[0.1, 0.1], [0.1, 0.1]])  # 2 years, 2 paths

        cumulative = calculate_cumulative_returns(returns)

        # Expected: (1.1 * 1.1) - 1 = 0.21
        expected = np.array([[0.1, 0.1], [0.21, 0.21]])

        np.testing.assert_array_almost_equal(cumulative, expected)

    def test_calculate_annualized_returns(self):
        """Test calculating annualized returns."""
        # Test case: 21% cumulative return over 2 years
        cumulative_returns = np.array([[0.1, 0.1], [0.21, 0.21]])  # 2 years, 2 paths

        annualized = calculate_annualized_returns(cumulative_returns, 2)

        # Expected: (1.21)^(1/2) - 1 ≈ 0.1
        expected = np.array([0.1, 0.1])

        np.testing.assert_array_almost_equal(annualized, expected, decimal=3)


class TestValidationFunctions:
    """Test validation functions."""

    def test_validate_returns_statistics_valid(self):
        """Test validation with valid statistics."""
        # Create returns with known statistics
        returns = np.random.normal(0.07, 0.18, (1, 1000, 10000))  # Large sample
        expected_returns = np.array([0.07])
        volatilities = np.array([0.18])

        is_valid, message = validate_returns_statistics(
            returns, expected_returns, volatilities, tolerance=0.01
        )

        assert is_valid
        assert message == "All statistics within tolerance"

    def test_validate_returns_statistics_invalid_mean(self):
        """Test validation with invalid mean."""
        # Create returns with wrong mean
        returns = np.random.normal(0.05, 0.18, (1, 1000, 1000))  # Wrong mean
        expected_returns = np.array([0.07])
        volatilities = np.array([0.18])

        is_valid, message = validate_returns_statistics(
            returns, expected_returns, volatilities, tolerance=0.01
        )

        assert not is_valid
        assert "mean" in message.lower()

    def test_validate_returns_statistics_invalid_std(self):
        """Test validation with invalid standard deviation."""
        # Create returns with wrong std
        returns = np.random.normal(0.07, 0.15, (1, 1000, 1000))  # Wrong std
        expected_returns = np.array([0.07])
        volatilities = np.array([0.18])

        is_valid, message = validate_returns_statistics(
            returns, expected_returns, volatilities, tolerance=0.01
        )

        assert not is_valid
        assert "std" in message.lower()


class TestIntegration:
    """Integration tests for the complete workflow."""

    def test_end_to_end_simulation(self):
        """Test complete end-to-end simulation workflow."""
        # Create configuration
        assets = create_default_portfolio()
        config = RandomReturnsConfig(
            asset_classes=assets,
            years=30,
            num_paths=1000,
            distribution="normal",
            seed=42,
        )

        # Generate returns
        generator = RandomReturnsGenerator(config)
        asset_returns = generator.generate_returns()

        # Calculate portfolio returns
        weights = generator.get_asset_weights()
        portfolio_returns = calculate_portfolio_returns(asset_returns, weights)

        # Calculate cumulative returns
        cumulative_returns = calculate_cumulative_returns(portfolio_returns)

        # Calculate annualized returns
        annualized_returns = calculate_annualized_returns(cumulative_returns, 30)

        # Verify results
        assert asset_returns.shape == (2, 30, 1000)
        assert portfolio_returns.shape == (30, 1000)
        assert cumulative_returns.shape == (30, 1000)
        assert annualized_returns.shape == (1000,)

        # Check that all values are finite
        assert np.all(np.isfinite(asset_returns))
        assert np.all(np.isfinite(portfolio_returns))
        assert np.all(np.isfinite(cumulative_returns))
        assert np.all(np.isfinite(annualized_returns))

        # Check that cumulative returns are reasonable (can decrease due to negative returns)
        for path in range(1000):
            path_cumulative = cumulative_returns[:, path]
            # Cumulative returns should be >= -1 (can't lose more than 100%)
            assert np.all(path_cumulative >= -1.0)
            # Should be finite
            assert np.all(np.isfinite(path_cumulative))

    def test_large_scale_simulation(self):
        """Test simulation with larger parameters."""
        assets = create_three_asset_portfolio()
        config = RandomReturnsConfig(
            asset_classes=assets,
            years=50,
            num_paths=10000,
            distribution="lognormal",
            seed=123,
        )

        generator = RandomReturnsGenerator(config)
        returns = generator.generate_returns()

        # Verify shape and basic properties
        assert returns.shape == (3, 50, 10000)
        assert np.all(np.isfinite(returns))
        assert np.all(returns >= -1.0)  # Can't lose more than 100%

        # Check that statistics are reasonable
        for i in range(3):
            asset_returns = returns[i].flatten()
            mean_return = np.mean(asset_returns)
            std_return = np.std(asset_returns)

            # Should be within reasonable bounds
            assert -0.5 < mean_return < 0.5  # Annual returns between -50% and 50%
            assert 0 < std_return < 1.0  # Volatility between 0% and 100%
