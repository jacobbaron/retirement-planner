"""
Portfolio evolution module for Monte Carlo simulation.

This module provides utilities for simulating portfolio evolution over time with
annual rebalancing to maintain target asset allocation.
"""

from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, field_validator

from .random_returns import AssetClass, RandomReturnsConfig, RandomReturnsGenerator


class PortfolioEvolutionConfig(BaseModel):
    """Configuration for portfolio evolution simulation."""

    model_config = {"arbitrary_types_allowed": True}

    initial_balance: float = Field(..., gt=0, description="Initial portfolio balance")
    asset_classes: List[AssetClass] = Field(
        ..., min_length=1, description="Asset classes in portfolio"
    )
    years: int = Field(..., ge=1, le=100, description="Number of years to simulate")
    num_paths: int = Field(
        ..., ge=1, le=100000, description="Number of simulation paths"
    )
    rebalance_threshold: float = Field(
        default=0.05, ge=0, le=1, description="Rebalancing threshold (drift tolerance)"
    )
    transaction_cost_rate: float = Field(
        default=0.0, ge=0, le=0.1, description="Transaction cost rate (0-10%)"
    )
    enable_rebalancing: bool = Field(
        default=True, description="Enable annual rebalancing"
    )
    returns_config: Optional[RandomReturnsConfig] = Field(
        default=None, description="Returns generation configuration"
    )

    @field_validator("asset_classes")
    @classmethod
    def validate_weights(cls, v: List[AssetClass]) -> List[AssetClass]:
        """Validate that portfolio weights sum to approximately 1.0."""
        total_weight = sum(asset.weight for asset in v)
        if not np.isclose(total_weight, 1.0, atol=1e-6):
            raise ValueError(f"Portfolio weights must sum to 1.0, got {total_weight}")
        return v


class PortfolioEvolutionResult(BaseModel):
    """Result of portfolio evolution simulation."""

    model_config = {"arbitrary_types_allowed": True}

    # Portfolio balances over time (years, num_paths)
    portfolio_balances: NDArray[np.float64] = Field(
        ..., description="Portfolio balances over time"
    )
    # Asset balances over time (num_assets, years, num_paths)
    asset_balances: NDArray[np.float64] = Field(
        ..., description="Asset balances over time"
    )
    # Rebalancing events (years, num_paths) - boolean array
    rebalancing_events: NDArray[np.bool_] = Field(
        ..., description="Rebalancing events over time"
    )
    # Transaction costs (years, num_paths)
    transaction_costs: NDArray[np.float64] = Field(
        ..., description="Transaction costs over time"
    )
    # Portfolio returns (years, num_paths)
    portfolio_returns: NDArray[np.float64] = Field(
        ..., description="Portfolio returns over time"
    )
    # Target weights (num_assets,)
    target_weights: NDArray[np.float64] = Field(..., description="Target asset weights")
    # Asset names
    asset_names: List[str] = Field(..., description="Asset class names")


class PortfolioEvolution:
    """Simulates portfolio evolution with annual rebalancing."""

    def __init__(self, config: PortfolioEvolutionConfig):
        """Initialize the portfolio evolution simulator.

        Args:
            config: Configuration for portfolio evolution
        """
        self.config = config
        self._validate_config()

        # Create returns generator if not provided
        if self.config.returns_config is None:
            self.returns_config = RandomReturnsConfig(
                asset_classes=self.config.asset_classes,
                years=self.config.years,
                num_paths=self.config.num_paths,
            )
        else:
            self.returns_config = self.config.returns_config

        self.returns_generator = RandomReturnsGenerator(self.returns_config)

    def _validate_config(self) -> None:
        """Validate the configuration parameters."""
        if self.config.initial_balance <= 0:
            raise ValueError("Initial balance must be positive")
        if self.config.years <= 0:
            raise ValueError("Number of years must be positive")
        if self.config.num_paths <= 0:
            raise ValueError("Number of paths must be positive")
        if len(self.config.asset_classes) == 0:
            raise ValueError("At least one asset class is required")

    def simulate(self) -> PortfolioEvolutionResult:
        """
        Simulate portfolio evolution over time.

        Returns:
            PortfolioEvolutionResult containing all simulation results
        """
        # Generate random returns
        asset_returns = self.returns_generator.generate_returns()

        # Get target weights
        target_weights = self.returns_generator.get_asset_weights()
        asset_names = self.returns_generator.get_asset_names()

        # Initialize arrays
        num_assets = len(self.config.asset_classes)
        years = self.config.years
        num_paths = self.config.num_paths

        # Portfolio balances over time (years, num_paths)
        portfolio_balances = np.zeros((years, num_paths))

        # Asset balances over time (num_assets, years, num_paths)
        asset_balances = np.zeros((num_assets, years, num_paths))

        # Rebalancing events (years, num_paths)
        rebalancing_events = np.zeros((years, num_paths), dtype=bool)

        # Transaction costs (years, num_paths)
        transaction_costs = np.zeros((years, num_paths))

        # Portfolio returns (years, num_paths)
        portfolio_returns = np.zeros((years, num_paths))

        # Initialize with initial balance
        portfolio_balances[0, :] = self.config.initial_balance

        # Initialize asset balances based on target weights
        for i in range(num_assets):
            asset_balances[i, 0, :] = self.config.initial_balance * target_weights[i]

        # Simulate year by year
        for year in range(years):
            if year == 0:
                # First year: no returns applied yet, just initial allocation
                continue

            # Apply returns to asset balances
            for asset_idx in range(num_assets):
                asset_balances[asset_idx, year, :] = asset_balances[
                    asset_idx, year - 1, :
                ] * (1 + asset_returns[asset_idx, year - 1, :])

            # Calculate total portfolio balance
            portfolio_balances[year, :] = np.sum(asset_balances[:, year, :], axis=0)

            # Calculate portfolio returns
            portfolio_returns[year - 1, :] = (
                portfolio_balances[year, :] / portfolio_balances[year - 1, :] - 1
            )

            # Check for rebalancing
            if self.config.enable_rebalancing:
                rebalance_needed, transaction_cost = self._check_rebalancing(
                    asset_balances[:, year, :],
                    portfolio_balances[year, :],
                    target_weights,
                )

                rebalancing_events[year, :] = rebalance_needed
                transaction_costs[year, :] = transaction_cost

                # Apply rebalancing if needed
                for path_idx in range(num_paths):
                    if rebalance_needed[path_idx]:
                        asset_balances[:, year, path_idx] = self._rebalance_portfolio(
                            asset_balances[:, year, path_idx],
                            portfolio_balances[year, path_idx],
                            target_weights,
                            transaction_cost[path_idx],
                        )

        return PortfolioEvolutionResult(
            portfolio_balances=portfolio_balances,
            asset_balances=asset_balances,
            rebalancing_events=rebalancing_events,
            transaction_costs=transaction_costs,
            portfolio_returns=portfolio_returns,
            target_weights=target_weights,
            asset_names=asset_names,
        )

    def _check_rebalancing(
        self,
        asset_balances: NDArray[np.float64],
        portfolio_balance: NDArray[np.float64],
        target_weights: NDArray[np.float64],
    ) -> Tuple[NDArray[np.bool_], NDArray[np.float64]]:
        """
        Check if rebalancing is needed based on drift from target weights.

        Args:
            asset_balances: Current asset balances (num_assets, num_paths)
            portfolio_balance: Current portfolio balance (num_paths,)
            target_weights: Target asset weights (num_assets,)

        Returns:
            Tuple of (rebalance_needed, transaction_cost)
        """
        num_paths = asset_balances.shape[1]
        rebalance_needed = np.zeros(num_paths, dtype=bool)
        transaction_cost = np.zeros(num_paths)

        for path_idx in range(num_paths):
            if portfolio_balance[path_idx] <= 0:
                # Portfolio is depleted, no rebalancing needed
                continue

            # Calculate current weights
            current_weights = asset_balances[:, path_idx] / portfolio_balance[path_idx]

            # Check if any asset has drifted beyond threshold
            max_drift = np.max(np.abs(current_weights - target_weights))

            if max_drift > self.config.rebalance_threshold:
                rebalance_needed[path_idx] = True

                # Calculate transaction cost based on rebalancing amount
                target_balances = portfolio_balance[path_idx] * target_weights
                rebalance_amount = np.sum(
                    np.abs(target_balances - asset_balances[:, path_idx])
                )
                transaction_cost[path_idx] = (
                    rebalance_amount * self.config.transaction_cost_rate
                )

        return rebalance_needed, transaction_cost

    def _rebalance_portfolio(
        self,
        asset_balances: NDArray[np.float64],
        portfolio_balance: float,
        target_weights: NDArray[np.float64],
        transaction_cost: float,
    ) -> NDArray[np.float64]:
        """
        Rebalance portfolio to target weights.

        Args:
            asset_balances: Current asset balances (num_assets,)
            portfolio_balance: Current portfolio balance
            target_weights: Target asset weights (num_assets,)
            transaction_cost: Transaction cost for rebalancing

        Returns:
            Rebalanced asset balances (num_assets,)
        """
        # Calculate target balances
        target_balances = portfolio_balance * target_weights

        # Apply transaction cost by reducing portfolio balance
        if transaction_cost > 0:
            # Reduce target balances proportionally to account for transaction cost
            cost_ratio = transaction_cost / portfolio_balance
            target_balances = target_balances * (1 - cost_ratio)

        return target_balances

    def get_target_weights(self) -> NDArray[np.float64]:
        """Get target asset weights."""
        return self.returns_generator.get_asset_weights()

    def get_asset_names(self) -> List[str]:
        """Get asset class names."""
        return self.returns_generator.get_asset_names()


def validate_rebalancing_accuracy(
    result: PortfolioEvolutionResult,
    tolerance: float = 1e-6,
) -> Tuple[bool, str]:
    """
    Validate that rebalancing restores target weights within tolerance.

    Args:
        result: Portfolio evolution result
        tolerance: Tolerance for weight validation

    Returns:
        Tuple of (is_valid, message)
    """
    num_assets, years, num_paths = result.asset_balances.shape

    for year in range(years):
        for path_idx in range(num_paths):
            portfolio_balance = result.portfolio_balances[year, path_idx]

            if portfolio_balance <= 0:
                continue  # Skip depleted portfolios

            # Calculate actual weights
            actual_weights = (
                result.asset_balances[:, year, path_idx] / portfolio_balance
            )

            # Check if weights are close to target
            weight_diff = np.abs(actual_weights - result.target_weights)
            max_diff = np.max(weight_diff)

            # Only validate rebalancing accuracy if rebalancing occurred in this year
            # or if this is the initial year (year 0)
            if year == 0 or result.rebalancing_events[year, path_idx]:
                if max_diff > tolerance:
                    return (
                        False,
                        f"Year {year}, Path {path_idx}: max weight drift {max_diff:.6f} exceeds tolerance {tolerance:.6f}",
                    )

    return True, "All rebalancing within tolerance"


def calculate_portfolio_statistics(
    result: PortfolioEvolutionResult,
) -> dict:
    """
    Calculate portfolio statistics from evolution results.

    Args:
        result: Portfolio evolution result

    Returns:
        Dictionary containing portfolio statistics
    """
    # Final portfolio balances
    final_balances = result.portfolio_balances[-1, :]

    # Portfolio returns over time
    portfolio_returns = result.portfolio_returns

    # Rebalancing frequency
    rebalancing_frequency = np.mean(result.rebalancing_events)

    # Total transaction costs
    total_transaction_costs = np.sum(result.transaction_costs, axis=0)

    # Portfolio volatility (annualized)
    portfolio_volatility = np.std(portfolio_returns, axis=0)

    # Portfolio mean return (annualized)
    portfolio_mean_return = np.mean(portfolio_returns, axis=0)

    return {
        "final_balance_mean": np.mean(final_balances),
        "final_balance_std": np.std(final_balances),
        "final_balance_p5": np.percentile(final_balances, 5),
        "final_balance_p50": np.percentile(final_balances, 50),
        "final_balance_p95": np.percentile(final_balances, 95),
        "rebalancing_frequency": rebalancing_frequency,
        "total_transaction_costs_mean": np.mean(total_transaction_costs),
        "total_transaction_costs_std": np.std(total_transaction_costs),
        "portfolio_volatility_mean": np.mean(portfolio_volatility),
        "portfolio_volatility_std": np.std(portfolio_volatility),
        "portfolio_mean_return_mean": np.mean(portfolio_mean_return),
        "portfolio_mean_return_std": np.std(portfolio_mean_return),
    }


def compare_rebalancing_strategies(
    config: PortfolioEvolutionConfig,
    rebalance_thresholds: List[float],
) -> dict:
    """
    Compare different rebalancing strategies.

    Args:
        config: Base portfolio evolution configuration
        rebalance_thresholds: List of rebalancing thresholds to compare

    Returns:
        Dictionary containing comparison results
    """
    results = {}

    for threshold in rebalance_thresholds:
        # Create config with this threshold
        test_config = config.model_copy()
        test_config.rebalance_threshold = threshold

        # Run simulation
        evolution = PortfolioEvolution(test_config)
        result = evolution.simulate()

        # Calculate statistics
        stats = calculate_portfolio_statistics(result)
        results[f"threshold_{threshold}"] = stats

    return results
