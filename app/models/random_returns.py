"""
Random returns generator for Monte Carlo simulation.

This module provides utilities for generating random returns for different asset classes
using normal and lognormal distributions with deterministic seeding for reproducible results.
"""

from typing import List, Literal, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, ValidationInfo, field_validator


class AssetClass(BaseModel):
    """Configuration for an asset class in the portfolio."""

    name: str = Field(..., description="Name of the asset class")
    expected_return: float = Field(
        ..., ge=-1, le=2, description="Expected annual return (decimal)"
    )
    volatility: float = Field(
        ..., ge=0, le=2, description="Annual volatility (decimal)"
    )
    weight: float = Field(..., ge=0, le=1, description="Portfolio weight (0-1)")


class RandomReturnsConfig(BaseModel):
    """Configuration for random returns generation."""

    model_config = {"arbitrary_types_allowed": True}

    asset_classes: List[AssetClass] = Field(
        ..., min_length=1, description="Asset classes in portfolio"
    )
    years: int = Field(..., ge=1, le=100, description="Number of years to simulate")
    num_paths: int = Field(
        ..., ge=1, le=100000, description="Number of simulation paths"
    )
    distribution: Literal["normal", "lognormal"] = Field(
        default="normal", description="Distribution type for returns"
    )
    correlation_matrix: Optional[NDArray[np.float64]] = Field(
        default=None, description="Correlation matrix between asset classes"
    )
    seed: Optional[int] = Field(
        default=None, ge=0, description="Random seed for reproducibility"
    )

    @field_validator("asset_classes")
    @classmethod
    def validate_weights(cls, v: List[AssetClass]) -> List[AssetClass]:
        """Validate that portfolio weights sum to approximately 1.0."""
        total_weight = sum(asset.weight for asset in v)
        if not np.isclose(total_weight, 1.0, atol=1e-6):
            raise ValueError(f"Portfolio weights must sum to 1.0, got {total_weight}")
        return v

    @field_validator("correlation_matrix")
    @classmethod
    def validate_correlation_matrix(
        cls, v: Optional[NDArray[np.float64]], info: ValidationInfo
    ) -> Optional[NDArray[np.float64]]:
        """Validate correlation matrix if provided."""
        if v is None:
            return v

        # Get asset classes from the model data
        asset_classes = info.data.get("asset_classes", [])
        num_assets = len(asset_classes)

        if v.shape != (num_assets, num_assets):
            raise ValueError(
                f"Correlation matrix must be {num_assets}x{num_assets}, got {v.shape}"
            )

        # Check if matrix is symmetric
        if not np.allclose(v, v.T, atol=1e-10):
            raise ValueError("Correlation matrix must be symmetric")

        # Check if diagonal elements are 1.0
        if not np.allclose(np.diag(v), 1.0, atol=1e-10):
            raise ValueError("Correlation matrix diagonal elements must be 1.0")

        # Check if matrix is positive definite (required for Cholesky decomposition)
        try:
            np.linalg.cholesky(v)
        except np.linalg.LinAlgError:
            raise ValueError("Correlation matrix must be positive definite")

        # Check if all correlations are in [-1, 1]
        if not np.all((v >= -1.0) & (v <= 1.0)):
            raise ValueError("All correlation values must be between -1 and 1")

        return v


class RandomReturnsGenerator:
    """Generates random returns for Monte Carlo simulation."""

    def __init__(self, config: RandomReturnsConfig):
        """Initialize the random returns generator.

        Args:
            config: Configuration for returns generation
        """
        self.config = config
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate the configuration parameters."""
        if self.config.num_paths <= 0:
            raise ValueError("Number of paths must be positive")
        if self.config.years <= 0:
            raise ValueError("Number of years must be positive")
        if len(self.config.asset_classes) == 0:
            raise ValueError("At least one asset class is required")

    def generate_returns(self) -> NDArray[np.float64]:
        """
        Generate random returns for all asset classes.

        Returns:
            Array of shape (num_assets, years, num_paths) containing returns
        """
        # Set random seed for reproducibility
        if self.config.seed is not None:
            np.random.seed(self.config.seed)

        num_assets = len(self.config.asset_classes)
        years = self.config.years
        num_paths = self.config.num_paths

        # Initialize returns array
        returns = np.zeros((num_assets, years, num_paths))

        if self.config.correlation_matrix is not None:
            # Generate correlated returns using Cholesky decomposition
            returns = self._generate_correlated_returns()
        else:
            # Generate independent returns (original behavior)
            returns = self._generate_independent_returns()

        return returns

    def _generate_independent_returns(self) -> NDArray[np.float64]:
        """Generate independent returns for each asset class."""
        num_assets = len(self.config.asset_classes)
        years = self.config.years
        num_paths = self.config.num_paths

        returns = np.zeros((num_assets, years, num_paths))

        for i, asset in enumerate(self.config.asset_classes):
            if self.config.distribution == "normal":
                # Generate normal returns
                returns[i] = np.random.normal(
                    asset.expected_return, asset.volatility, (years, num_paths)
                )
            elif self.config.distribution == "lognormal":
                # Generate lognormal returns
                # For lognormal: if X ~ N(mu, sigma), then exp(X) ~ LogNormal(mu, sigma)
                # We want returns with mean = expected_return and std = volatility
                # So we need to adjust the parameters
                mu = np.log(1 + asset.expected_return) - 0.5 * asset.volatility**2
                sigma = asset.volatility

                log_returns = np.random.normal(mu, sigma, (years, num_paths))
                returns[i] = np.exp(log_returns) - 1  # Convert to simple returns
            else:
                raise ValueError(
                    f"Unsupported distribution: {self.config.distribution}"
                )

        return returns

    def _generate_correlated_returns(self) -> NDArray[np.float64]:
        """Generate correlated returns using Cholesky decomposition."""
        num_assets = len(self.config.asset_classes)
        years = self.config.years
        num_paths = self.config.num_paths

        # Get correlation matrix and volatilities
        corr_matrix = self.config.correlation_matrix
        if corr_matrix is None:
            raise ValueError(
                "Correlation matrix is required for correlated returns generation"
            )

        volatilities = self.get_volatilities()
        expected_returns = self.get_expected_returns()

        # Create covariance matrix from correlation matrix and volatilities
        # Cov[i,j] = Corr[i,j] * Vol[i] * Vol[j]
        cov_matrix = np.outer(volatilities, volatilities) * corr_matrix

        # Perform Cholesky decomposition
        try:
            cholesky_matrix = np.linalg.cholesky(cov_matrix)
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Failed to perform Cholesky decomposition: {e}")

        # Generate independent standard normal random variables
        # Shape: (num_assets, years, num_paths)
        independent_randoms = np.random.standard_normal((num_assets, years, num_paths))

        # Apply Cholesky transformation to create correlated random variables
        # For each time step and path, transform independent variables
        correlated_randoms = np.zeros_like(independent_randoms)

        for t in range(years):
            for p in range(num_paths):
                # Get independent random vector for this time/path
                z = independent_randoms[:, t, p]  # Shape: (num_assets,)

                # Apply Cholesky transformation: Y = L * Z
                y = cholesky_matrix @ z  # Shape: (num_assets,)

                # Add expected returns
                correlated_randoms[:, t, p] = expected_returns + y

        # Convert to appropriate distribution if needed
        if self.config.distribution == "lognormal":
            # For lognormal, we need to adjust the parameters
            # The correlated_randoms are already in log space with correct mean/variance
            returns = np.zeros_like(correlated_randoms)
            for i, asset in enumerate(self.config.asset_classes):
                # Adjust for lognormal: mu = log(1 + expected_return) - 0.5 * volatility^2
                mu = np.log(1 + asset.expected_return) - 0.5 * asset.volatility**2
                # The Cholesky transformation already gives us the right variance
                # We just need to adjust the mean
                returns[i] = mu + (correlated_randoms[i] - expected_returns[i])
                # Convert to simple returns
                returns[i] = np.exp(returns[i]) - 1
        else:
            # For normal distribution, use the correlated randoms directly
            returns = correlated_randoms

        return returns

    def get_asset_names(self) -> List[str]:
        """Get list of asset class names."""
        return [asset.name for asset in self.config.asset_classes]

    def get_asset_weights(self) -> NDArray[np.float64]:
        """Get array of asset weights."""
        return np.array([asset.weight for asset in self.config.asset_classes])

    def get_expected_returns(self) -> NDArray[np.float64]:
        """Get array of expected returns."""
        return np.array([asset.expected_return for asset in self.config.asset_classes])

    def get_volatilities(self) -> NDArray[np.float64]:
        """Get array of volatilities."""
        return np.array([asset.volatility for asset in self.config.asset_classes])


def create_default_portfolio() -> List[AssetClass]:
    """Create a default 60/40 stock/bond portfolio."""
    return [
        AssetClass(
            name="Stocks",
            expected_return=0.07,  # 7% expected return
            volatility=0.18,  # 18% volatility
            weight=0.6,
        ),
        AssetClass(
            name="Bonds",
            expected_return=0.03,  # 3% expected return
            volatility=0.06,  # 6% volatility
            weight=0.4,
        ),
    ]


def create_three_asset_portfolio() -> List[AssetClass]:
    """Create a three-asset portfolio (stocks, bonds, REITs)."""
    return [
        AssetClass(name="Stocks", expected_return=0.07, volatility=0.18, weight=0.5),
        AssetClass(name="Bonds", expected_return=0.03, volatility=0.06, weight=0.3),
        AssetClass(name="REITs", expected_return=0.06, volatility=0.15, weight=0.2),
    ]


def create_default_correlation_matrix(num_assets: int = 2) -> NDArray[np.float64]:
    """Create a default correlation matrix for common asset classes.

    Args:
        num_assets: Number of assets (2 for stocks/bonds, 3 for stocks/bonds/REITs)

    Returns:
        Correlation matrix
    """
    if num_assets == 2:
        # Typical stocks/bonds correlation
        return np.array([[1.0, -0.2], [-0.2, 1.0]])
    elif num_assets == 3:
        # Typical stocks/bonds/REITs correlation
        return np.array(
            [
                [1.0, -0.2, 0.6],  # Stocks vs Bonds, Stocks vs REITs
                [-0.2, 1.0, 0.1],  # Bonds vs Stocks, Bonds vs REITs
                [0.6, 0.1, 1.0],  # REITs vs Stocks, REITs vs Bonds
            ]
        )
    else:
        # For other sizes, create a reasonable default
        corr_matrix = np.eye(num_assets)
        # Add some typical correlations
        for i in range(num_assets):
            for j in range(i + 1, num_assets):
                # Default correlation of 0.3 for different asset classes
                corr_matrix[i, j] = 0.3
                corr_matrix[j, i] = 0.3
        return corr_matrix


def validate_correlation_accuracy(
    returns: NDArray[np.float64],
    target_correlation: NDArray[np.float64],
    tolerance: float = 0.05,
) -> Tuple[bool, str]:
    """
    Validate that generated returns have correlations close to target values.

    Args:
        returns: Generated returns array (num_assets, years, num_paths)
        target_correlation: Target correlation matrix
        tolerance: Tolerance for correlation validation

    Returns:
        Tuple of (is_valid, message)
    """
    num_assets = returns.shape[0]

    # Calculate empirical correlation matrix
    # Flatten across years and paths to get all observations
    empirical_corr = np.corrcoef(returns.reshape(num_assets, -1))

    # Check correlations are within tolerance
    for i in range(num_assets):
        for j in range(i + 1, num_assets):
            empirical_corr_ij = empirical_corr[i, j]
            target_corr_ij = target_correlation[i, j]
            diff = abs(empirical_corr_ij - target_corr_ij)

            if diff > tolerance:
                return (
                    False,
                    f"Correlation between assets {i} and {j}: "
                    f"empirical {empirical_corr_ij:.4f} differs from target "
                    f"{target_corr_ij:.4f} by {diff:.4f}",
                )

    return True, "All correlations within tolerance"


def validate_returns_statistics(
    returns: NDArray[np.float64],
    expected_returns: NDArray[np.float64],
    volatilities: NDArray[np.float64],
    tolerance: float = 0.01,
) -> Tuple[bool, str]:
    """
    Validate that generated returns have statistics close to expected values.

    Args:
        returns: Generated returns array (num_assets, years, num_paths)
        expected_returns: Expected returns for each asset
        volatilities: Expected volatilities for each asset
        tolerance: Tolerance for statistical validation

    Returns:
        Tuple of (is_valid, message)
    """
    num_assets = returns.shape[0]

    for i in range(num_assets):
        asset_returns = returns[i].flatten()

        # Calculate actual statistics
        actual_mean = np.mean(asset_returns)
        actual_std = np.std(asset_returns)

        # Check if within tolerance
        mean_diff = abs(actual_mean - expected_returns[i])
        std_diff = abs(actual_std - volatilities[i])

        if mean_diff > tolerance:
            return (
                False,
                f"Asset {i}: mean {actual_mean:.4f} differs from expected {expected_returns[i]:.4f} by {mean_diff:.4f}",
            )

        if std_diff > tolerance:
            return (
                False,
                f"Asset {i}: std {actual_std:.4f} differs from expected {volatilities[i]:.4f} by {std_diff:.4f}",
            )

    return True, "All statistics within tolerance"


def calculate_portfolio_returns(
    asset_returns: NDArray[np.float64], weights: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Calculate portfolio returns from individual asset returns.

    Args:
        asset_returns: Returns array (num_assets, years, num_paths)
        weights: Portfolio weights (num_assets,)

    Returns:
        Portfolio returns array (years, num_paths)
    """
    if asset_returns.shape[0] != len(weights):
        raise ValueError("Number of assets must match number of weights")

    # Weighted average of asset returns
    portfolio_returns: NDArray[np.float64] = np.sum(
        asset_returns * weights[:, np.newaxis, np.newaxis], axis=0
    )

    return portfolio_returns


def calculate_cumulative_returns(returns: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Calculate cumulative returns from period returns.

    Args:
        returns: Period returns array (years, num_paths)

    Returns:
        Cumulative returns array (years, num_paths)
    """
    # Convert to cumulative returns: (1 + r1) * (1 + r2) * ... * (1 + rn) - 1
    cumulative: NDArray[np.float64] = np.cumprod(1 + returns, axis=0) - 1
    return cumulative


def calculate_annualized_returns(
    returns: NDArray[np.float64], years: int
) -> NDArray[np.float64]:
    """
    Calculate annualized returns from cumulative returns.

    Args:
        returns: Cumulative returns array (years, num_paths)
        years: Number of years

    Returns:
        Annualized returns array (num_paths,)
    """
    if returns.shape[0] != years:
        raise ValueError("Returns array length must match number of years")

    # Annualized return = (1 + cumulative_return)^(1/years) - 1
    final_returns = returns[-1, :]  # Final cumulative returns
    annualized = (1 + final_returns) ** (1 / years) - 1

    return annualized
