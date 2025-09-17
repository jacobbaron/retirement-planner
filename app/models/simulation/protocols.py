"""
Protocol interfaces for simulation providers.

This module defines clean Protocol interfaces that enable dependency injection
and composable simulation architecture. Each protocol represents a specific
aspect of the retirement planning simulation.

Design Principles:
1. Single Responsibility: Each protocol has one clear purpose
2. Dependency Inversion: Depend on abstractions, not concrete implementations
3. Substitutability: Any implementation of a protocol can be swapped
4. Testability: Protocols enable easy mocking for unit tests
"""

from typing import Any, Dict, Optional, Protocol

import numpy as np
from numpy.typing import NDArray

from app.models.time_grid import UnitSystem


class ReturnsProvider(Protocol):
    """
    Provides market returns for assets over time.

    Generates correlated asset returns for Monte Carlo simulations,
    supporting various distributions and correlation structures.
    """

    def generate_returns(
        self, years: int, num_paths: int, seed: Optional[int] = None
    ) -> NDArray[np.float64]:
        """
        Generate asset returns for simulation.

        Args:
            years: Number of years to simulate
            num_paths: Number of Monte Carlo paths
            seed: Random seed for reproducibility

        Returns:
            NDArray of shape (num_assets, years, num_paths) with returns

        Raises:
            ValueError: If years or num_paths <= 0
        """
        ...

    def get_asset_names(self) -> list[str]:
        """Get names of assets in order."""
        ...

    def get_target_weights(self) -> NDArray[np.float64]:
        """Get target portfolio weights for assets."""
        ...

    def get_expected_returns(self) -> NDArray[np.float64]:
        """Get long-term expected returns for each asset."""
        ...


class IncomeProvider(Protocol):
    """
    Provides annual income from all sources.

    Integrates salary, business income, investment income, retirement benefits,
    Social Security, and other income sources with proper timing and growth.
    """

    def get_annual_income(self, year: int, path: int, unit_system: UnitSystem) -> float:
        """
        Get total annual income for a specific year and path.

        Args:
            year: Simulation year (0-based)
            path: Monte Carlo path index
            unit_system: For inflation adjustments and real/nominal conversion

        Returns:
            Total annual income (in real or nominal terms per unit_system)

        Raises:
            ValueError: If year or path are invalid
        """
        ...

    def get_income_breakdown(
        self, year: int, path: int, unit_system: UnitSystem
    ) -> Dict[str, float]:
        """
        Get detailed income breakdown by source.

        Returns:
            Dict mapping income source names to amounts
        """
        ...


class ExpenseProvider(Protocol):
    """
    Provides annual expenses including baseline and lumpy events.

    Handles all expense categories (housing, transportation, healthcare, etc.)
    with inflation adjustments and one-time lumpy expenses.
    """

    def get_annual_expenses(
        self, year: int, path: int, unit_system: UnitSystem
    ) -> float:
        """
        Get total annual expenses for a specific year and path.

        Args:
            year: Simulation year (0-based)
            path: Monte Carlo path index
            unit_system: For inflation adjustments

        Returns:
            Total annual expenses (real or nominal per unit_system)
        """
        ...

    def get_expense_breakdown(
        self, year: int, path: int, unit_system: UnitSystem
    ) -> Dict[str, float]:
        """
        Get detailed expense breakdown by category.

        Returns:
            Dict mapping expense categories to amounts
        """
        ...


class LiabilityProvider(Protocol):
    """
    Provides liability payments (mortgages, loans, etc.).

    Handles mortgage amortization, other loan payments, and liability
    management including refinancing scenarios.
    """

    def get_annual_payments(
        self, year: int, path: int, unit_system: UnitSystem
    ) -> float:
        """
        Get total annual liability payments.

        Args:
            year: Simulation year (0-based)
            path: Monte Carlo path index
            unit_system: For inflation adjustments

        Returns:
            Total annual payments (principal + interest)
        """
        ...

    def get_payment_breakdown(
        self, year: int, path: int, unit_system: UnitSystem
    ) -> Dict[str, Dict[str, float]]:
        """
        Get detailed payment breakdown by liability.

        Returns:
            Dict mapping liability names to payment details:
            {"mortgage_1": {"principal": 5000, "interest": 3000, "total": 8000}}
        """
        ...

    def get_remaining_balances(self, year: int, path: int) -> Dict[str, float]:
        """Get remaining balances for all liabilities."""
        ...


class WithdrawalPolicy(Protocol):
    """
    Determines withdrawal amounts from portfolio.

    Implements various withdrawal strategies (fixed real, fixed percentage,
    VPW, dynamic strategies) based on portfolio balance and other factors.
    """

    def compute_withdrawal(
        self,
        year: int,
        path: int,
        cash_need: float,
        portfolio_balance: float,
        unit_system: UnitSystem,
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Compute withdrawal amount for the year.

        Args:
            year: Simulation year (0-based)
            path: Monte Carlo path index
            cash_need: Net cash needed (expenses + liabilities - income)
            portfolio_balance: Current portfolio value
            unit_system: For inflation adjustments
            context: Additional context (returns history, success metrics, etc.)

        Returns:
            Withdrawal amount (>= 0)

        Raises:
            ValueError: If portfolio_balance < 0 or other invalid inputs
        """
        ...

    def get_strategy_name(self) -> str:
        """Get human-readable name of withdrawal strategy."""
        ...


class RebalancingStrategy(Protocol):
    """
    Handles portfolio rebalancing decisions and execution.

    Determines when and how to rebalance portfolio to target weights,
    including transaction cost calculations.
    """

    def should_rebalance(
        self,
        current_weights: NDArray[np.float64],
        target_weights: NDArray[np.float64],
        year: int,
        path: int,
    ) -> bool:
        """
        Determine if rebalancing is needed.

        Args:
            current_weights: Current portfolio weights by asset
            target_weights: Target portfolio weights
            year: Simulation year
            path: Monte Carlo path index

        Returns:
            True if rebalancing should occur
        """
        ...

    def compute_transaction_cost(
        self,
        current_balances: NDArray[np.float64],
        target_balances: NDArray[np.float64],
    ) -> float:
        """
        Compute transaction cost for rebalancing.

        Args:
            current_balances: Current asset balances
            target_balances: Target asset balances after rebalancing

        Returns:
            Total transaction cost
        """
        ...

    def get_strategy_name(self) -> str:
        """Get human-readable name of rebalancing strategy."""
        ...


class TaxCalculator(Protocol):
    """
    Calculates annual taxes on various income sources.

    Handles federal and state income taxes, capital gains, and other
    tax considerations. Can be a no-op for simulations without tax modeling.
    """

    def compute_annual_taxes(
        self,
        year: int,
        path: int,
        ordinary_income: float,
        capital_gains: float,
        withdrawals_by_account_type: Dict[str, float],
        unit_system: UnitSystem,
    ) -> float:
        """
        Compute total annual tax liability.

        Args:
            year: Simulation year (0-based)
            path: Monte Carlo path index
            ordinary_income: Total ordinary income (salary, interest, etc.)
            capital_gains: Realized capital gains
            withdrawals_by_account_type: Withdrawals by account type
                {"taxable": 1000, "traditional_401k": 5000, "roth_401k": 2000}
            unit_system: For inflation adjustments

        Returns:
            Total tax liability
        """
        ...

    def get_effective_tax_rate(
        self, year: int, path: int, total_income: float
    ) -> float:
        """Get effective tax rate for given income level."""
        ...


class PortfolioEngine(Protocol):
    """
    Manages portfolio state and evolution.

    Handles portfolio initialization, cash flows, return application,
    rebalancing, and state tracking across multiple Monte Carlo paths.
    """

    def initialize(
        self,
        initial_balance: float,
        target_weights: NDArray[np.float64],
        asset_names: list[str],
        num_paths: int,
    ) -> None:
        """
        Initialize portfolio for simulation.

        Args:
            initial_balance: Starting portfolio value
            target_weights: Target allocation weights
            asset_names: Names of assets in order
            num_paths: Number of Monte Carlo paths
        """
        ...

    def apply_cash_flow(self, amount: float, year: int, path: int) -> None:
        """
        Apply cash flow to portfolio (positive = contribution, negative = withdrawal).

        Args:
            amount: Cash flow amount
            year: Simulation year
            path: Monte Carlo path index
        """
        ...

    def apply_returns(
        self, asset_returns: NDArray[np.float64], year: int, path: int
    ) -> None:
        """
        Apply market returns to portfolio.

        Args:
            asset_returns: Returns for each asset for this year/path
            year: Simulation year
            path: Monte Carlo path index
        """
        ...

    def get_current_balance(self, year: int, path: int) -> float:
        """Get total portfolio balance for year/path."""
        ...

    def get_current_weights(self, year: int, path: int) -> NDArray[np.float64]:
        """Get current portfolio weights for year/path."""
        ...

    def get_asset_balances(self, year: int, path: int) -> NDArray[np.float64]:
        """Get individual asset balances for year/path."""
        ...

    def apply_rebalancing(
        self,
        target_weights: NDArray[np.float64],
        transaction_cost: float,
        year: int,
        path: int,
    ) -> None:
        """
        Rebalance portfolio to target weights.

        Args:
            target_weights: Target allocation weights
            transaction_cost: Cost of rebalancing
            year: Simulation year
            path: Monte Carlo path index
        """
        ...


# Type aliases for common data structures
AssetReturns = NDArray[np.float64]  # Shape: (num_assets, years, num_paths)
PortfolioBalances = NDArray[np.float64]  # Shape: (years, num_paths)
AssetBalances = NDArray[np.float64]  # Shape: (num_assets, years, num_paths)
CashFlows = NDArray[np.float64]  # Shape: (years, num_paths)
