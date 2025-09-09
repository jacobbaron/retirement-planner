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
    create_default_correlation_matrix,
    create_default_portfolio,
    create_three_asset_portfolio,
    validate_correlation_accuracy,
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

    def test_valid_correlation_matrix(self):
        """Test creating config with valid correlation matrix."""
        assets = create_default_portfolio()
        corr_matrix = create_default_correlation_matrix(2)

        config = RandomReturnsConfig(
            asset_classes=assets,
            years=30,
            num_paths=1000,
            correlation_matrix=corr_matrix,
        )
        assert config.correlation_matrix is not None
        np.testing.assert_array_equal(config.correlation_matrix, corr_matrix)

    def test_invalid_correlation_matrix_size(self):
        """Test validation of correlation matrix size."""
        assets = create_default_portfolio()  # 2 assets
        # Wrong size correlation matrix (3x3 instead of 2x2)
        invalid_corr = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.2], [0.3, 0.2, 1.0]])

        with pytest.raises(ValueError, match="Correlation matrix must be 2x2"):
            RandomReturnsConfig(
                asset_classes=assets,
                years=30,
                num_paths=1000,
                correlation_matrix=invalid_corr,
            )

    def test_invalid_correlation_matrix_symmetric(self):
        """Test validation of correlation matrix symmetry."""
        assets = create_default_portfolio()
        # Non-symmetric matrix
        invalid_corr = np.array([[1.0, 0.5], [0.3, 1.0]])  # 0.5 != 0.3

        with pytest.raises(ValueError, match="Correlation matrix must be symmetric"):
            RandomReturnsConfig(
                asset_classes=assets,
                years=30,
                num_paths=1000,
                correlation_matrix=invalid_corr,
            )

    def test_invalid_correlation_matrix_diagonal(self):
        """Test validation of correlation matrix diagonal elements."""
        assets = create_default_portfolio()
        # Diagonal not equal to 1.0
        invalid_corr = np.array([[1.0, 0.5], [0.5, 0.9]])  # 0.9 != 1.0

        with pytest.raises(
            ValueError, match="Correlation matrix diagonal elements must be 1.0"
        ):
            RandomReturnsConfig(
                asset_classes=assets,
                years=30,
                num_paths=1000,
                correlation_matrix=invalid_corr,
            )

    def test_invalid_correlation_matrix_positive_definite(self):
        """Test validation of correlation matrix positive definiteness."""
        assets = create_default_portfolio()
        # Not positive definite (e.g., correlation > 1)
        invalid_corr = np.array([[1.0, 1.1], [1.1, 1.0]])  # Correlation > 1

        with pytest.raises(
            ValueError, match="Correlation matrix must be positive definite"
        ):
            RandomReturnsConfig(
                asset_classes=assets,
                years=30,
                num_paths=1000,
                correlation_matrix=invalid_corr,
            )

    def test_invalid_correlation_matrix_bounds(self):
        """Test validation of correlation matrix bounds."""
        assets = create_default_portfolio()
        # Correlation outside [-1, 1]
        invalid_corr = np.array([[1.0, 1.5], [1.5, 1.0]])  # Correlation > 1

        with pytest.raises(
            ValueError, match="Correlation matrix must be positive definite"
        ):
            RandomReturnsConfig(
                asset_classes=assets,
                years=30,
                num_paths=1000,
                correlation_matrix=invalid_corr,
            )


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

    def test_generate_correlated_normal_returns(self):
        """Test generating correlated normal returns."""
        assets = create_default_portfolio()
        corr_matrix = create_default_correlation_matrix(2)

        config = RandomReturnsConfig(
            asset_classes=assets,
            years=10,
            num_paths=10000,  # Large sample for correlation testing
            distribution="normal",
            correlation_matrix=corr_matrix,
            seed=42,
        )
        generator = RandomReturnsGenerator(config)
        returns = generator.generate_returns()

        # Check shape
        assert returns.shape == (2, 10, 10000)

        # Check that returns are reasonable
        assert np.all(np.isfinite(returns))

        # Check statistics are close to expected values
        is_valid, message = validate_returns_statistics(
            returns,
            generator.get_expected_returns(),
            generator.get_volatilities(),
            tolerance=0.05,
        )
        assert is_valid, message

        # Check correlations are close to target
        is_corr_valid, corr_message = validate_correlation_accuracy(
            returns, corr_matrix, tolerance=0.05
        )
        assert is_corr_valid, corr_message

    def test_generate_correlated_lognormal_returns(self):
        """Test generating correlated lognormal returns."""
        assets = create_default_portfolio()
        corr_matrix = create_default_correlation_matrix(2)

        config = RandomReturnsConfig(
            asset_classes=assets,
            years=10,
            num_paths=10000,
            distribution="lognormal",
            correlation_matrix=corr_matrix,
            seed=42,
        )
        generator = RandomReturnsGenerator(config)
        returns = generator.generate_returns()

        # Check shape
        assert returns.shape == (2, 10, 10000)

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

        # Check correlations are close to target
        is_corr_valid, corr_message = validate_correlation_accuracy(
            returns, corr_matrix, tolerance=0.05
        )
        assert is_corr_valid, corr_message

    def test_three_asset_correlated_returns(self):
        """Test generating correlated returns for three assets."""
        assets = create_three_asset_portfolio()
        corr_matrix = create_default_correlation_matrix(3)

        config = RandomReturnsConfig(
            asset_classes=assets,
            years=5,
            num_paths=10000,
            distribution="normal",
            correlation_matrix=corr_matrix,
            seed=123,
        )
        generator = RandomReturnsGenerator(config)
        returns = generator.generate_returns()

        # Check shape
        assert returns.shape == (3, 5, 10000)

        # Check that returns are reasonable
        assert np.all(np.isfinite(returns))

        # Check correlations are close to target
        is_corr_valid, corr_message = validate_correlation_accuracy(
            returns, corr_matrix, tolerance=0.05
        )
        assert is_corr_valid, corr_message

    def test_correlated_returns_deterministic_seeding(self):
        """Test that correlated returns are reproducible with same seed."""
        assets = create_default_portfolio()
        corr_matrix = create_default_correlation_matrix(2)

        config = RandomReturnsConfig(
            asset_classes=assets,
            years=5,
            num_paths=100,
            correlation_matrix=corr_matrix,
            seed=456,
        )

        generator1 = RandomReturnsGenerator(config)
        returns1 = generator1.generate_returns()

        generator2 = RandomReturnsGenerator(config)
        returns2 = generator2.generate_returns()

        # Results should be identical with same seed
        np.testing.assert_array_equal(returns1, returns2)

    def test_correlated_returns_different_seeds(self):
        """Test that different seeds produce different correlated results."""
        assets = create_default_portfolio()
        corr_matrix = create_default_correlation_matrix(2)

        config1 = RandomReturnsConfig(
            asset_classes=assets,
            years=5,
            num_paths=100,
            correlation_matrix=corr_matrix,
            seed=123,
        )

        config2 = RandomReturnsConfig(
            asset_classes=assets,
            years=5,
            num_paths=100,
            correlation_matrix=corr_matrix,
            seed=456,
        )

        generator1 = RandomReturnsGenerator(config1)
        returns1 = generator1.generate_returns()

        generator2 = RandomReturnsGenerator(config2)
        returns2 = generator2.generate_returns()

        # Results should be different with different seeds
        assert not np.array_equal(returns1, returns2)

        # But correlations should still be similar
        is_corr_valid1, _ = validate_correlation_accuracy(
            returns1, corr_matrix, tolerance=0.1
        )
        is_corr_valid2, _ = validate_correlation_accuracy(
            returns2, corr_matrix, tolerance=0.1
        )
        assert is_corr_valid1
        assert is_corr_valid2


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


class TestCorrelationFunctions:
    """Test correlation utility functions."""

    def test_create_default_correlation_matrix_two_assets(self):
        """Test creating default correlation matrix for two assets."""
        corr_matrix = create_default_correlation_matrix(2)

        expected = np.array([[1.0, -0.2], [-0.2, 1.0]])
        np.testing.assert_array_equal(corr_matrix, expected)

    def test_create_default_correlation_matrix_three_assets(self):
        """Test creating default correlation matrix for three assets."""
        corr_matrix = create_default_correlation_matrix(3)

        expected = np.array([[1.0, -0.2, 0.6], [-0.2, 1.0, 0.1], [0.6, 0.1, 1.0]])
        np.testing.assert_array_equal(corr_matrix, expected)

    def test_create_default_correlation_matrix_four_assets(self):
        """Test creating default correlation matrix for four assets."""
        corr_matrix = create_default_correlation_matrix(4)

        # Should be 4x4 with 1.0 on diagonal and 0.3 off-diagonal
        assert corr_matrix.shape == (4, 4)
        np.testing.assert_array_equal(np.diag(corr_matrix), np.ones(4))

        # Check off-diagonal elements are 0.3
        for i in range(4):
            for j in range(4):
                if i != j:
                    assert corr_matrix[i, j] == 0.3

    def test_validate_correlation_accuracy_valid(self):
        """Test correlation validation with valid correlations."""
        # Create returns with known correlation
        np.random.seed(42)
        n_obs = 10000

        # Generate correlated data
        corr_target = 0.5
        x = np.random.standard_normal(n_obs)
        y = corr_target * x + np.sqrt(1 - corr_target**2) * np.random.standard_normal(
            n_obs
        )

        # Reshape to match expected format (num_assets, years, num_paths)
        returns = np.array([x, y]).reshape(2, 1, n_obs)
        target_corr = np.array([[1.0, corr_target], [corr_target, 1.0]])

        is_valid, message = validate_correlation_accuracy(
            returns, target_corr, tolerance=0.1
        )

        assert is_valid
        assert "within tolerance" in message

    def test_validate_correlation_accuracy_invalid(self):
        """Test correlation validation with invalid correlations."""
        # Create returns with wrong correlation
        np.random.seed(42)
        n_obs = 1000

        # Generate uncorrelated data
        x = np.random.standard_normal(n_obs)
        y = np.random.standard_normal(n_obs)

        # Reshape to match expected format
        returns = np.array([x, y]).reshape(2, 1, n_obs)
        target_corr = np.array([[1.0, 0.8], [0.8, 1.0]])  # High target correlation

        is_valid, message = validate_correlation_accuracy(
            returns, target_corr, tolerance=0.1
        )

        assert not is_valid
        assert "differs from target" in message


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

    def test_end_to_end_correlated_simulation(self):
        """Test complete end-to-end correlated simulation workflow."""
        # Create configuration with correlation
        assets = create_default_portfolio()
        corr_matrix = create_default_correlation_matrix(2)

        config = RandomReturnsConfig(
            asset_classes=assets,
            years=30,
            num_paths=10000,
            distribution="normal",
            correlation_matrix=corr_matrix,
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
        assert asset_returns.shape == (2, 30, 10000)
        assert portfolio_returns.shape == (30, 10000)
        assert cumulative_returns.shape == (30, 10000)
        assert annualized_returns.shape == (10000,)

        # Check that all values are finite
        assert np.all(np.isfinite(asset_returns))
        assert np.all(np.isfinite(portfolio_returns))
        assert np.all(np.isfinite(cumulative_returns))
        assert np.all(np.isfinite(annualized_returns))

        # Check that correlations are close to target
        is_corr_valid, corr_message = validate_correlation_accuracy(
            asset_returns, corr_matrix, tolerance=0.05
        )
        assert is_corr_valid, corr_message

        # Check that cumulative returns are reasonable
        for path in range(10000):
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
