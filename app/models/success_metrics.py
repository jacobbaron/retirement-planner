"""
Success metrics module for Monte Carlo simulation.

This module provides comprehensive success metrics and percentile analysis
for retirement planning simulations, including success rates, terminal wealth
percentiles, and first failure year distributions.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from .withdrawal_rules import WithdrawalResult


class SuccessMetricsConfig(BaseModel):
    """Configuration for success metrics calculation."""

    model_config = {"arbitrary_types_allowed": True}

    confidence_levels: List[float] = Field(
        default=[0.05, 0.25, 0.50, 0.75, 0.95],
        description="Confidence levels for percentile calculations",
    )
    failure_threshold: float = Field(
        default=0.0, ge=0, description="Minimum balance threshold for failure"
    )
    success_threshold: float = Field(
        default=0.0, ge=0, description="Minimum balance threshold for success"
    )


class SuccessMetricsResult(BaseModel):
    """Result of success metrics calculation."""

    model_config = {"arbitrary_types_allowed": True}

    # Success rate (fraction of paths that never fail)
    success_rate: float = Field(..., description="Success rate (0-1)")

    # Terminal wealth percentiles
    terminal_wealth_percentiles: Dict[str, float] = Field(
        ..., description="Terminal wealth percentiles"
    )

    # First failure year distribution
    first_failure_year_stats: Dict[str, float] = Field(
        ..., description="First failure year statistics"
    )

    # Portfolio balance statistics over time
    balance_statistics: Dict[str, NDArray[np.float64]] = Field(
        ..., description="Portfolio balance statistics over time"
    )

    # Withdrawal statistics
    withdrawal_statistics: Dict[str, float] = Field(
        ..., description="Withdrawal statistics"
    )

    # Risk metrics
    risk_metrics: Dict[str, float] = Field(
        ..., description="Risk metrics (volatility, max drawdown, etc.)"
    )


class SuccessMetricsCalculator:
    """Calculator for comprehensive success metrics."""

    def __init__(self, config: Optional[SuccessMetricsConfig] = None):
        """Initialize the success metrics calculator.

        Args:
            config: Configuration for success metrics calculation
        """
        self.config = config or SuccessMetricsConfig()

    def calculate_metrics(
        self,
        withdrawal_result: WithdrawalResult,
        portfolio_returns: Optional[NDArray[np.float64]] = None,
    ) -> SuccessMetricsResult:
        """
        Calculate comprehensive success metrics from withdrawal simulation results.

        Args:
            withdrawal_result: Withdrawal simulation result
            portfolio_returns: Portfolio returns over time (optional)

        Returns:
            SuccessMetricsResult containing all calculated metrics
        """
        # Calculate success rate
        success_rate = self._calculate_success_rate(withdrawal_result)

        # Calculate terminal wealth percentiles
        terminal_wealth_percentiles = self._calculate_terminal_wealth_percentiles(
            withdrawal_result
        )

        # Calculate first failure year distribution
        first_failure_year_stats = self._calculate_first_failure_year_stats(
            withdrawal_result
        )

        # Calculate portfolio balance statistics over time
        balance_statistics = self._calculate_balance_statistics(withdrawal_result)

        # Calculate withdrawal statistics
        withdrawal_statistics = self._calculate_withdrawal_statistics(withdrawal_result)

        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(
            withdrawal_result, portfolio_returns
        )

        return SuccessMetricsResult(
            success_rate=success_rate,
            terminal_wealth_percentiles=terminal_wealth_percentiles,
            first_failure_year_stats=first_failure_year_stats,
            balance_statistics=balance_statistics,
            withdrawal_statistics=withdrawal_statistics,
            risk_metrics=risk_metrics,
        )

    def _calculate_success_rate(self, result: WithdrawalResult) -> float:
        """Calculate success rate (fraction of paths that never fail)."""
        return result.success_rate

    def _calculate_terminal_wealth_percentiles(
        self, result: WithdrawalResult
    ) -> Dict[str, float]:
        """Calculate terminal wealth percentiles."""
        final_balances = result.portfolio_balances[-1, :]

        percentiles = {}
        for level in self.config.confidence_levels:
            percentile_key = f"p{int(level * 100)}"
            percentiles[percentile_key] = np.percentile(final_balances, level * 100)

        return percentiles

    def _calculate_first_failure_year_stats(
        self, result: WithdrawalResult
    ) -> Dict[str, float]:
        """Calculate first failure year distribution statistics."""
        failure_years = result.first_failure_years[result.first_failure_years != -1]

        if len(failure_years) == 0:
            return {
                "mean": -1.0,
                "std": 0.0,
                "min": -1.0,
                "max": -1.0,
                "count": 0.0,
            }

        return {
            "mean": float(np.mean(failure_years)),
            "std": float(np.std(failure_years)),
            "min": float(np.min(failure_years)),
            "max": float(np.max(failure_years)),
            "count": float(len(failure_years)),
        }

    def _calculate_balance_statistics(
        self, result: WithdrawalResult
    ) -> Dict[str, NDArray[np.float64]]:
        """Calculate portfolio balance statistics over time."""
        years, num_paths = result.portfolio_balances.shape

        # Calculate statistics for each year
        balance_mean = np.zeros(years)
        balance_std = np.zeros(years)
        balance_p5 = np.zeros(years)
        balance_p50 = np.zeros(years)
        balance_p95 = np.zeros(years)

        for year in range(years):
            year_balances = result.portfolio_balances[year, :]
            balance_mean[year] = np.mean(year_balances)
            balance_std[year] = np.std(year_balances)
            balance_p5[year] = np.percentile(year_balances, 5)
            balance_p50[year] = np.percentile(year_balances, 50)
            balance_p95[year] = np.percentile(year_balances, 95)

        return {
            "mean": balance_mean,
            "std": balance_std,
            "p5": balance_p5,
            "p50": balance_p50,
            "p95": balance_p95,
        }

    def _calculate_withdrawal_statistics(
        self, result: WithdrawalResult
    ) -> Dict[str, float]:
        """Calculate withdrawal statistics."""
        # Total withdrawals per path
        total_withdrawals = np.sum(result.withdrawals, axis=0)

        # Average annual withdrawal per path
        avg_annual_withdrawal = np.mean(result.withdrawals, axis=0)

        # Withdrawal volatility per path
        withdrawal_volatility = np.std(result.withdrawals, axis=0)

        return {
            "total_withdrawals_mean": float(np.mean(total_withdrawals)),
            "total_withdrawals_std": float(np.std(total_withdrawals)),
            "total_withdrawals_p5": float(np.percentile(total_withdrawals, 5)),
            "total_withdrawals_p50": float(np.percentile(total_withdrawals, 50)),
            "total_withdrawals_p95": float(np.percentile(total_withdrawals, 95)),
            "avg_annual_withdrawal_mean": float(np.mean(avg_annual_withdrawal)),
            "avg_annual_withdrawal_std": float(np.std(avg_annual_withdrawal)),
            "withdrawal_volatility_mean": float(np.mean(withdrawal_volatility)),
            "withdrawal_volatility_std": float(np.std(withdrawal_volatility)),
        }

    def _calculate_risk_metrics(
        self,
        result: WithdrawalResult,
        portfolio_returns: Optional[NDArray[np.float64]] = None,
    ) -> Dict[str, float]:
        """Calculate risk metrics."""
        # Portfolio volatility (if returns are available)
        portfolio_volatility = 0.0
        if portfolio_returns is not None:
            portfolio_volatility = float(np.mean(np.std(portfolio_returns, axis=0)))

        # Maximum drawdown
        max_drawdown = self._calculate_max_drawdown(result.portfolio_balances)

        # Value at Risk (VaR) - 5th percentile of final balances
        final_balances = result.portfolio_balances[-1, :]
        var_5 = float(np.percentile(final_balances, 5))

        # Conditional Value at Risk (CVaR) - expected value below VaR
        cvar_5 = float(np.mean(final_balances[final_balances <= var_5]))

        return {
            "portfolio_volatility": portfolio_volatility,
            "max_drawdown": max_drawdown,
            "var_5": var_5,
            "cvar_5": cvar_5,
            "failure_rate": 1.0 - result.success_rate,
        }

    def _calculate_max_drawdown(self, portfolio_balances: NDArray[np.float64]) -> float:
        """Calculate maximum drawdown from peak."""
        years, num_paths = portfolio_balances.shape
        max_drawdown = 0.0

        for path_idx in range(num_paths):
            path_balances = portfolio_balances[:, path_idx]
            peak = path_balances[0]  # Start with initial balance as peak
            path_max_drawdown = 0.0

            for year in range(years):
                if path_balances[year] > peak:
                    peak = path_balances[year]
                else:
                    drawdown = (peak - path_balances[year]) / peak
                    path_max_drawdown = max(path_max_drawdown, drawdown)

            max_drawdown = max(max_drawdown, path_max_drawdown)

        return float(max_drawdown)


def compare_success_metrics(
    metrics_results: List[Tuple[str, SuccessMetricsResult]]
) -> Dict[str, Dict[str, float]]:
    """
    Compare success metrics across different strategies.

    Args:
        metrics_results: List of (strategy_name, SuccessMetricsResult) tuples

    Returns:
        Dictionary containing comparison results
    """
    comparison = {}

    for strategy_name, result in metrics_results:
        comparison[strategy_name] = {
            "success_rate": result.success_rate,
            "terminal_wealth_p5": result.terminal_wealth_percentiles["p5"],
            "terminal_wealth_p50": result.terminal_wealth_percentiles["p50"],
            "terminal_wealth_p95": result.terminal_wealth_percentiles["p95"],
            "first_failure_year_mean": result.first_failure_year_stats["mean"],
            "max_drawdown": result.risk_metrics["max_drawdown"],
            "var_5": result.risk_metrics["var_5"],
        }

    return comparison


def generate_success_report(
    result: SuccessMetricsResult, strategy_name: str = "Strategy"
) -> str:
    """
    Generate a human-readable success report.

    Args:
        result: Success metrics result
        strategy_name: Name of the strategy

    Returns:
        Formatted success report string
    """
    report = f"""
=== {strategy_name} Success Report ===

Success Rate: {result.success_rate:.1%}

Terminal Wealth Percentiles:
  P5:  ${result.terminal_wealth_percentiles['p5']:,.0f}
  P50: ${result.terminal_wealth_percentiles['p50']:,.0f}
  P95: ${result.terminal_wealth_percentiles['p95']:,.0f}

First Failure Year Statistics:
  Mean: {result.first_failure_year_stats['mean']:.1f}
  Std:  {result.first_failure_year_stats['std']:.1f}
  Count: {result.first_failure_year_stats['count']:.0f}

Risk Metrics:
  Max Drawdown: {result.risk_metrics['max_drawdown']:.1%}
  VaR (5%): ${result.risk_metrics['var_5']:,.0f}
  CVaR (5%): ${result.risk_metrics['cvar_5']:,.0f}
  Portfolio Volatility: {result.risk_metrics['portfolio_volatility']:.1%}

Withdrawal Statistics:
  Total Withdrawals (Mean): ${result.withdrawal_statistics['total_withdrawals_mean']:,.0f}
  Avg Annual Withdrawal: ${result.withdrawal_statistics['avg_annual_withdrawal_mean']:,.0f}
"""

    return report


def validate_success_metrics(
    result: SuccessMetricsResult,
    expected_properties: Dict[str, float],
    tolerance: float = 1e-6,
) -> Tuple[bool, str]:
    """
    Validate success metrics against expected properties.

    Args:
        result: Success metrics result
        expected_properties: Expected properties to validate
        tolerance: Tolerance for validation

    Returns:
        Tuple of (is_valid, message)
    """
    # Flatten the result into a single dictionary for validation
    flat_result = {
        "success_rate": result.success_rate,
        **result.terminal_wealth_percentiles,
        **result.first_failure_year_stats,
        **result.withdrawal_statistics,
        **result.risk_metrics,
    }

    # Validate expected properties
    for property_name, expected_value in expected_properties.items():
        if property_name not in flat_result:
            return False, f"Property {property_name} not found in metrics"

        actual_value = flat_result[property_name]
        if not np.isclose(actual_value, expected_value, atol=tolerance):
            return (
                False,
                f"Property {property_name}: expected {expected_value}, got {actual_value}",
            )

    return True, "All properties within tolerance"
