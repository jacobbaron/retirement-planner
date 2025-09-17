"""
Simulation orchestration module.

This module provides the flexible simulation orchestrator that coordinates
all retirement planning components (income, expenses, portfolio, withdrawals,
taxes, etc.) through clean Protocol interfaces.

Key Components:
- protocols: Protocol interfaces for all simulation providers
- config: Pydantic configuration model for simulation parameters
- result: Comprehensive result model for simulation outputs
- orchestrator: Main orchestrator that coordinates all providers
- adapters: Adapters that wrap existing engines with Protocol interfaces
- factory: Factory methods to create configurations from scenarios
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Import protocols for type checking
    from .protocols import (
        ExpenseProvider,
        IncomeProvider,
        LiabilityProvider,
        PortfolioEngine,
        RebalancingStrategy,
        ReturnsProvider,
        TaxCalculator,
        WithdrawalPolicy,
    )

__all__ = [
    "ReturnsProvider",
    "IncomeProvider",
    "ExpenseProvider",
    "LiabilityProvider",
    "WithdrawalPolicy",
    "RebalancingStrategy",
    "TaxCalculator",
    "PortfolioEngine",
]
