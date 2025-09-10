"""
Withdrawal rules module for Monte Carlo simulation.

This module provides various withdrawal strategies for retirement planning,
including the 4% rule, fixed percentage, and Variable Percentage Withdrawal (VPW).
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from .portfolio_evolution import PortfolioEvolutionResult


class WithdrawalRuleConfig(BaseModel):
    """Base configuration for withdrawal rules."""

    model_config = {"arbitrary_types_allowed": True}

    initial_balance: float = Field(..., gt=0, description="Initial portfolio balance")
    years: int = Field(..., ge=1, le=100, description="Number of years to simulate")
    num_paths: int = Field(
        ..., ge=1, le=100000, description="Number of simulation paths"
    )
    inflation_rate: float = Field(
        default=0.03, ge=0, le=0.2, description="Annual inflation rate"
    )


class WithdrawalResult(BaseModel):
    """Result of withdrawal rule simulation."""

    model_config = {"arbitrary_types_allowed": True}

    # Withdrawal amounts over time (years, num_paths)
    withdrawals: NDArray[np.float64] = Field(
        ..., description="Withdrawal amounts over time"
    )
    # Portfolio balances after withdrawals (years, num_paths)
    portfolio_balances: NDArray[np.float64] = Field(
        ..., description="Portfolio balances after withdrawals"
    )
    # Failure events (years, num_paths) - boolean array
    failures: NDArray[np.bool_] = Field(
        ..., description="Failure events (insufficient funds)"
    )
    # First failure year for each path (-1 if no failure)
    first_failure_years: NDArray[np.int32] = Field(
        ..., description="First failure year for each path"
    )
    # Success rate (fraction of paths that never fail)
    success_rate: float = Field(..., description="Success rate (0-1)")


class WithdrawalRule(ABC):
    """Abstract base class for withdrawal rules."""

    def __init__(self, config: WithdrawalRuleConfig):
        """Initialize the withdrawal rule.

        Args:
            config: Configuration for the withdrawal rule
        """
        self.config = config

    @abstractmethod
    def calculate_withdrawal(
        self,
        year: int,
        portfolio_balance: float,
        path_idx: int,
        portfolio_returns: Optional[NDArray[np.float64]] = None,
    ) -> float:
        """
        Calculate withdrawal amount for a given year and portfolio balance.

        Args:
            year: Current year (0-based)
            portfolio_balance: Current portfolio balance
            path_idx: Path index for this simulation
            portfolio_returns: Portfolio returns over time (optional)

        Returns:
            Withdrawal amount
        """

    def apply_withdrawals(
        self, portfolio_result: PortfolioEvolutionResult
    ) -> WithdrawalResult:
        """
        Apply withdrawal rule to portfolio evolution results.

        Args:
            portfolio_result: Portfolio evolution results

        Returns:
            WithdrawalResult containing withdrawal simulation results
        """
        years = self.config.years
        num_paths = self.config.num_paths

        # Initialize arrays
        withdrawals = np.zeros((years, num_paths))
        portfolio_balances = np.zeros((years, num_paths))
        failures = np.zeros((years, num_paths), dtype=bool)
        first_failure_years = np.full(num_paths, -1, dtype=np.int32)

        # Copy initial portfolio balances
        portfolio_balances[0, :] = portfolio_result.portfolio_balances[0, :]

        # Apply withdrawals year by year
        for year in range(years):
            for path_idx in range(num_paths):
                current_balance = portfolio_balances[year, path_idx]

                # Skip if portfolio is already depleted
                if current_balance <= 0:
                    failures[year, path_idx] = True
                    if first_failure_years[path_idx] == -1:
                        first_failure_years[path_idx] = year
                    continue

                # Calculate withdrawal amount
                withdrawal = self.calculate_withdrawal(
                    year, current_balance, path_idx, portfolio_result.portfolio_returns
                )

                # Check if withdrawal exceeds available balance
                if withdrawal > current_balance:
                    # Withdraw all available balance
                    withdrawals[year, path_idx] = current_balance
                    portfolio_balances[year, path_idx] = 0.0
                    failures[year, path_idx] = True
                    if first_failure_years[path_idx] == -1:
                        first_failure_years[path_idx] = year
                else:
                    # Normal withdrawal
                    withdrawals[year, path_idx] = withdrawal
                    portfolio_balances[year, path_idx] = current_balance - withdrawal

            # Apply portfolio returns for next year (if not the last year)
            if year < years - 1:
                for path_idx in range(num_paths):
                    if portfolio_balances[year, path_idx] > 0:
                        # Apply return for next year
                        if year < portfolio_result.portfolio_returns.shape[0]:
                            return_rate = portfolio_result.portfolio_returns[
                                year, path_idx
                            ]
                            portfolio_balances[year + 1, path_idx] = portfolio_balances[
                                year, path_idx
                            ] * (1 + return_rate)
                        else:
                            # No more returns available, keep balance
                            portfolio_balances[year + 1, path_idx] = portfolio_balances[
                                year, path_idx
                            ]

        # Calculate success rate
        success_rate = np.mean(first_failure_years == -1)

        return WithdrawalResult(
            withdrawals=withdrawals,
            portfolio_balances=portfolio_balances,
            failures=failures,
            first_failure_years=first_failure_years,
            success_rate=success_rate,
        )


class FixedRealWithdrawalRule(WithdrawalRule):
    """4% real withdrawal rule (initial percentage, then inflation-adjusted)."""

    def __init__(
        self, config: WithdrawalRuleConfig, initial_withdrawal_rate: float = 0.04
    ):
        """Initialize the fixed real withdrawal rule.

        Args:
            config: Configuration for the withdrawal rule
            initial_withdrawal_rate: Initial withdrawal rate (default 4%)
        """
        super().__init__(config)
        self.initial_withdrawal_rate = initial_withdrawal_rate
        self.initial_withdrawal_amount = (
            config.initial_balance * initial_withdrawal_rate
        )

    def calculate_withdrawal(
        self,
        year: int,
        portfolio_balance: float,
        path_idx: int,
        portfolio_returns: Optional[NDArray[np.float64]] = None,
    ) -> float:
        """Calculate inflation-adjusted withdrawal amount."""
        # Apply inflation to initial withdrawal amount
        inflation_factor = (1 + self.config.inflation_rate) ** year
        return self.initial_withdrawal_amount * inflation_factor


class FixedPercentageWithdrawalRule(WithdrawalRule):
    """Fixed percentage of current balance withdrawal rule."""

    def __init__(self, config: WithdrawalRuleConfig, withdrawal_rate: float = 0.04):
        """Initialize the fixed percentage withdrawal rule.

        Args:
            config: Configuration for the withdrawal rule
            withdrawal_rate: Withdrawal rate as percentage of current balance
        """
        super().__init__(config)
        self.withdrawal_rate = withdrawal_rate

    def calculate_withdrawal(
        self,
        year: int,
        portfolio_balance: float,
        path_idx: int,
        portfolio_returns: Optional[NDArray[np.float64]] = None,
    ) -> float:
        """Calculate withdrawal as fixed percentage of current balance."""
        return portfolio_balance * self.withdrawal_rate


class VPWWithdrawalRule(WithdrawalRule):
    """Variable Percentage Withdrawal (VPW) rule based on age and remaining years."""

    def __init__(
        self,
        config: WithdrawalRuleConfig,
        initial_age: int = 65,
        max_age: int = 100,
        base_withdrawal_rate: float = 0.04,
    ):
        """Initialize the VPW withdrawal rule.

        Args:
            config: Configuration for the withdrawal rule
            initial_age: Initial age at retirement
            max_age: Maximum expected age
            base_withdrawal_rate: Base withdrawal rate at initial age
        """
        super().__init__(config)
        self.initial_age = initial_age
        self.max_age = max_age
        self.base_withdrawal_rate = base_withdrawal_rate

        # Calculate VPW rates for each year
        self.vpw_rates = self._calculate_vpw_rates()

    def _calculate_vpw_rates(self) -> NDArray[np.float64]:
        """Calculate VPW rates for each year based on remaining years."""
        rates = np.zeros(self.config.years)

        for year in range(self.config.years):
            current_age = self.initial_age + year
            remaining_years = max(1, self.max_age - current_age)

            # VPW rate increases as remaining years decrease
            # Simple linear increase from base rate to 1.0 at max age
            if remaining_years <= 1:
                rates[year] = 1.0  # Withdraw everything in final year
            else:
                # Linear interpolation from base rate to 1.0
                max_remaining = self.max_age - self.initial_age
                progress = (max_remaining - remaining_years) / max_remaining
                rates[year] = self.base_withdrawal_rate + progress * (
                    1.0 - self.base_withdrawal_rate
                )

        return rates

    def calculate_withdrawal(
        self,
        year: int,
        portfolio_balance: float,
        path_idx: int,
        portfolio_returns: Optional[NDArray[np.float64]] = None,
    ) -> float:
        """Calculate VPW withdrawal amount."""
        if year >= len(self.vpw_rates):
            # Beyond expected lifetime, withdraw everything
            return portfolio_balance

        return float(portfolio_balance * self.vpw_rates[year])


def calculate_withdrawal_statistics(result: WithdrawalResult) -> Dict[str, float]:
    """
    Calculate statistics from withdrawal simulation results.

    Args:
        result: Withdrawal simulation result

    Returns:
        Dictionary containing withdrawal statistics
    """
    # Final portfolio balances
    final_balances = result.portfolio_balances[-1, :]

    # Total withdrawals over time
    total_withdrawals = np.sum(result.withdrawals, axis=0)

    # Average annual withdrawal
    avg_annual_withdrawal = np.mean(result.withdrawals, axis=0)

    # Withdrawal volatility
    withdrawal_volatility = np.std(result.withdrawals, axis=0)

    # Failure statistics
    failure_years = result.first_failure_years[result.first_failure_years != -1]

    return {
        "success_rate": result.success_rate,
        "final_balance_mean": np.mean(final_balances),
        "final_balance_std": np.std(final_balances),
        "final_balance_p5": np.percentile(final_balances, 5),
        "final_balance_p50": np.percentile(final_balances, 50),
        "final_balance_p95": np.percentile(final_balances, 95),
        "total_withdrawals_mean": np.mean(total_withdrawals),
        "total_withdrawals_std": np.std(total_withdrawals),
        "avg_annual_withdrawal_mean": np.mean(avg_annual_withdrawal),
        "avg_annual_withdrawal_std": np.std(avg_annual_withdrawal),
        "withdrawal_volatility_mean": np.mean(withdrawal_volatility),
        "withdrawal_volatility_std": np.std(withdrawal_volatility),
        "first_failure_year_mean": (
            np.mean(failure_years) if len(failure_years) > 0 else -1
        ),
        "first_failure_year_std": (
            np.std(failure_years) if len(failure_years) > 0 else 0
        ),
        "failure_rate": 1.0 - result.success_rate,
    }


def compare_withdrawal_strategies(
    portfolio_result: PortfolioEvolutionResult,
    withdrawal_rules: List[WithdrawalRule],
) -> Dict[str, Dict[str, float]]:
    """
    Compare different withdrawal strategies.

    Args:
        portfolio_result: Portfolio evolution results
        withdrawal_rules: List of withdrawal rules to compare

    Returns:
        Dictionary containing comparison results for each strategy
    """
    results = {}

    for i, rule in enumerate(withdrawal_rules):
        # Apply withdrawal rule
        withdrawal_result = rule.apply_withdrawals(portfolio_result)

        # Calculate statistics
        stats = calculate_withdrawal_statistics(withdrawal_result)
        results[f"strategy_{i}"] = stats

    return results


def validate_withdrawal_rule(
    rule: WithdrawalRule,
    portfolio_result: PortfolioEvolutionResult,
    expected_properties: Dict[str, float],
    tolerance: float = 1e-6,
) -> Tuple[bool, str]:
    """
    Validate withdrawal rule against expected properties.

    Args:
        rule: Withdrawal rule to validate
        portfolio_result: Portfolio evolution results
        expected_properties: Expected properties to validate
        tolerance: Tolerance for validation

    Returns:
        Tuple of (is_valid, message)
    """
    # Apply withdrawal rule
    result = rule.apply_withdrawals(portfolio_result)

    # Calculate statistics
    stats = calculate_withdrawal_statistics(result)

    # Validate expected properties
    for property_name, expected_value in expected_properties.items():
        if property_name not in stats:
            return False, f"Property {property_name} not found in statistics"

        actual_value = stats[property_name]
        if not np.isclose(actual_value, expected_value, atol=tolerance):
            return (
                False,
                f"Property {property_name}: expected {expected_value}, got {actual_value}",
            )

    return True, "All properties within tolerance"
